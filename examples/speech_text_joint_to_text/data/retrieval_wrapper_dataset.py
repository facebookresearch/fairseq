# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from random import randint
from typing import Dict, List, Tuple

import torch

from fairseq.data.audio.speech_to_text_joint_dataset import SpeechToTextJointDatasetItem
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data import data_utils as fairseq_data_utils


logger = logging.getLogger(__name__)

CONFIGS = {
    "num_negative": 5,
    "max_words": 5
}


def split_into_words(samples, src_dict):
    words = []
    for x in samples:
        assert x.src_txt_tokens.size(0) > 1, f"Impossible to process empty sample {x}"
        sample_words = []
        start_word_idx = 0
        for ph_idx in range(1, x.src_txt_tokens.size(0)):
            if src_dict[x.src_txt_tokens[ph_idx]].startswith("\u2581"):
                sample_words.append((start_word_idx, ph_idx))
                start_word_idx = ph_idx
        sample_words.append((start_word_idx, ph_idx))
        words.append(sample_words)
    return words


class SpeechTextRetrievalDataset(BaseWrapperDataset):
    def __init__(self, dataset, dictionary):
        super().__init__(dataset)
        self.dictionary = dictionary
        self.__indexes_to_ignore = set()

    def filter_short_utterances(self, indices, min_len, ignore_invalid_inputs=False):
        ignored = indices[self.sizes[indices] < min_len]
        indices = indices[self.sizes[indices] >= min_len]
        num_ignored = len(ignored)
        if num_ignored > 0:
            if not ignore_invalid_inputs:
                raise Exception("{:,} samples are too short and will be skipped, min_len={}".format(
                    num_ignored, min_len))
            logger.warning("{:,} samples are too short and will be skipped, min_len={}".format(
                num_ignored, min_len))
        self.__indexes_to_ignore = set(ignored)
        return indices

    @staticmethod
    def is_contained(a: torch.Tensor, b: torch.Tensor) -> bool:
        if b.shape[0] < a.shape[0]:
            return False
        return (a == b.unfold(0, a.shape[0], 1)).all(dim=-1).any()

    @staticmethod
    def take_random_sample(words: List[Tuple[int, int]], target: torch.Tensor) -> torch.Tensor:
        start = randint(0, len(words)-1)
        ln = randint(0, min(CONFIGS['max_words'], len(words) - start) - 1)
        return target[words[start][0]:words[start+ln][1]]

    def in_batch_negative_sample(self, batch_size, sample_id):
        # With one one sample in the batch it is not possible to
        # find negative samples
        if batch_size < 2:
            return -1
        # If there are only 2 samples, take the other one
        if batch_size == 2:
            return 0 if sample_id == 1 else 1
        # Take a random sample from the batch
        negative_sample = randint(0, batch_size - 1)
        n_tentative = 0
        while negative_sample == sample_id:
            negative_sample = randint(0, batch_size - 1)
            n_tentative += 1
            if n_tentative > 100:
                return -1
        return negative_sample

    def out_of_batch_negative_sample(self, sample_ds_id):
        # Take a random sample from the batch
        ds_max = len(self.dataset) - 1
        negative_sample_idx = randint(0, ds_max)
        n_tentative = 0
        while negative_sample_idx == sample_ds_id or negative_sample_idx in self.__indexes_to_ignore:
            negative_sample_idx = randint(0, ds_max)
            n_tentative += 1
            if n_tentative > 100:
                raise Exception(
                    f"unable to find negatives for {sample_ds_id} in the whole dataset.")
        negative_sample = self.dataset[negative_sample_idx]
        words = split_into_words([negative_sample], self.dictionary)[0]
        return self.take_random_sample(words, negative_sample.src_txt_tokens)

    def pick_negative(self, sample_id, words_samples, samples):
        batch_size = len(words_samples)
        n_tentative = 0
        while True:
            # take random sample in the batch
            negative_sample_idx = self.in_batch_negative_sample(batch_size, sample_id)
            # if no negative was found in batch, take it from other
            # samples in the dataset
            if negative_sample_idx < 0:
                tentative_neg = self.out_of_batch_negative_sample(samples[sample_id].index)
            else:
                tentative_neg = self.take_random_sample(
                    words_samples[negative_sample_idx], samples[negative_sample_idx].src_txt_tokens)
            # Ensure we have not sampled words that are present in the
            # current sentence as well
            if not self.is_contained(tentative_neg, samples[sample_id].src_txt_tokens):
                break
            n_tentative += 1
            if n_tentative > 100:
                raise Exception(
                    f"unable to find negatives for {self.dictionary.string(samples[sample_id].src_txt_tokens)}")
        return tentative_neg

    def sample_positive_and_negative_retrieval_items(
        self, samples: List[SpeechToTextJointDatasetItem]
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        # We are assuming sentencepiece-like segmentation
        # of source text (phonemes in our case)
        words_samples = split_into_words(samples, self.dictionary)
        positive_and_negatives = []
        for sample_id in range(len(words_samples)):
            positive_sample = self.take_random_sample(words_samples[sample_id], samples[sample_id].src_txt_tokens)
            negative_samples = []
            for _ in range(CONFIGS['num_negative']):
                negative_samples.append(self.pick_negative(sample_id, words_samples, samples))
            positive_and_negatives.append((positive_sample, negative_samples))
        return positive_and_negatives

    def collater(
        self, samples: List[SpeechToTextJointDatasetItem], return_order: bool = False
    ) -> Dict:
        collated_samples = self.dataset.collater(samples, return_order=True)
        if collated_samples == {}:
            return collated_samples
        positive_retrs, negative_retrs = None, None
        try:
            positive_and_negatives_retr = self.sample_positive_and_negative_retrieval_items(samples)
        except Exception as e:
            logger.error(f"Samples causing error are: {samples}")
            raise e

        positive_retrs = [fairseq_data_utils.collate_tokens(
            [pos for pos, _ in positive_and_negatives_retr],
            self.dictionary.pad(),
            self.dictionary.eos(),
        ).index_select(0, collated_samples['order'])]
        negative_retrs = [
            fairseq_data_utils.collate_tokens(
                [negs[i] for _, negs in positive_and_negatives_retr],
                self.dictionary.pad(),
                self.dictionary.eos(),
            ).index_select(0, collated_samples['order']) for i in range(CONFIGS['num_negative'])]
        collated_samples['net_input']['positive_retr_tokens'] = positive_retrs
        collated_samples['net_input']['negative_retr_tokens'] = negative_retrs
        if not return_order:
            del collated_samples["order"]
        return collated_samples
