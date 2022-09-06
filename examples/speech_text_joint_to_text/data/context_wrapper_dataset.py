# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from random import randint, random, sample, shuffle
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from fairseq.data import data_utils as fairseq_data_utils
from examples.speech_text_joint_to_text.data.retrieval_wrapper_dataset import SpeechTextRetrievalDataset


logger = logging.getLogger(__name__)


class SpeechToTextJointDatasetWithContextItem(NamedTuple):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    src_lang_tag: Optional[int] = None
    tgt_alignment: Optional[torch.Tensor] = None
    entities: Optional[torch.Tensor] = None
    context: Optional[List[torch.Tensor]] = None


def split_into_words(samples, src_dict):
    words = []
    for x in samples:
        assert x.target.size(0) > 1, f"Impossible to process empty sample {x}"
        sample_words = []
        start_word_idx = 0
        for ph_idx in range(1, x.target.size(0)):
            if src_dict[x.target[ph_idx]].startswith("\u2581"):
                sample_words.append((start_word_idx, ph_idx))
                start_word_idx = ph_idx
        sample_words.append((start_word_idx, ph_idx))
        words.append(sample_words)
    return words


class SpeechTextContextDataset(SpeechTextRetrievalDataset):
    CONTEXT_TAG = "#CONTEXT#"

    def __init__(self, dataset, dictionary, num_negatives=1, max_words=5, max_positives=3):
        super().__init__(dataset, dictionary, num_negatives, max_words)
        self.context_tag_idx = self.dictionary.index(SpeechTextContextDataset.CONTEXT_TAG)
        self.max_positives = max_positives

    def __getitem__(self, index):
        parent_item = super().__getitem__(index)
        positive_samples = []
        if random() < 0.8:
            if len(parent_item.entities) > 0 and random() < 0.9:
                num_to_pick = randint(0, min(len(parent_item.entities) - 1, self.max_positives))
                to_pick = sample(range(len(parent_item.entities)), num_to_pick)
                for i in to_pick:
                    positive_samples.append(parent_item.entities[i])
            else:
                positive_samples.append(
                    self.take_random_sample(split_into_words([parent_item], self.dictionary)[0], parent_item.target))
        target = parent_item.target
        for s in positive_samples:
            target = self.append_context_tag(target, s)
        if target is None:
            logger.error(f"Issue in detecting {positive_samples} in {self.dictionary.string(parent_item.target)}")
            raise Exception(f"Issue in detecting {positive_samples} in {parent_item.target}")
        return SpeechToTextJointDatasetWithContextItem(
            parent_item.index,
            parent_item.source,
            target,
            parent_item.src_txt_tokens,
            parent_item.tgt_lang_tag,
            parent_item.src_lang_tag,
            parent_item.tgt_alignment,
            parent_item.entities,
            positive_samples
        )

    def append_context_tag(self, target_tensor, entity):
        entity_len = entity.size(0)
        for i in range(target_tensor.size(0)):
            if torch.equal(target_tensor[i:i+entity_len], entity):
                return torch.cat((target_tensor[:i+entity_len], torch.LongTensor([self.context_tag_idx]), target_tensor[i+entity_len:]))

    def sample_negative_retrieval_items(
        self, samples: List[SpeechToTextJointDatasetWithContextItem]
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        # We are assuming sentencepiece-like segmentation
        # of source text (phonemes in our case)
        words_samples = split_into_words(samples, self.dictionary)
        negatives = []
        for sample_id in range(len(words_samples)):
            negative_samples = []
            for _ in range(self.num_negatives):
                negative_samples.append(self.pick_negative(sample_id, words_samples, samples))
            negatives.append((negative_samples))
        return negatives

    def collater(
        self, samples: List[SpeechToTextJointDatasetWithContextItem], return_order: bool = False
    ) -> Dict:
        collated_samples = self.dataset.collater(samples, return_order=True)
        if collated_samples == {}:
            return collated_samples
        try:
            negatives = self.sample_negative_retrieval_items(samples)
        except Exception as e:
            logger.error(f"Samples causing error are: {samples}")
            raise e

        contexts = []
        context_lengths = []
        for idx in collated_samples['order']:
            all_context = samples[idx].context + negatives[idx]
            shuffle(all_context)
            collated_tokens = fairseq_data_utils.collate_tokens(
                all_context,
                self.dictionary.pad(),
                self.dictionary.eos(),
            )
            extra_lens = 0
            if samples[idx].tgt_lang_tag is not None:
                tags = torch.LongTensor([samples[idx].tgt_lang_tag] * collated_tokens.shape[0]).unsqueeze(1)
                collated_tokens = torch.cat((tags, collated_tokens), dim=1)
                extra_lens = 1
            contexts.append(collated_tokens)
            context_lengths.append(torch.LongTensor([len(c) + extra_lens for c in all_context]))

        collated_samples['net_input']['context_list'] = contexts
        collated_samples['net_input']['context_lengths_list'] = context_lengths
        if not return_order:
            del collated_samples["order"]
        return collated_samples
