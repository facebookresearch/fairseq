#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, Iterator, List

import torch
from fairseq import utils
from omegaconf import open_dict
from torch import nn

from tqdm import tqdm

from fairseq.hub_utils import GeneratorHubInterface


logger = logging.getLogger(__name__)


class MultichannelGeneratorHubInterface(GeneratorHubInterface):
    """Pytorch Hub interface for generating sequences from a pre-trained
    multichannel language model.
    """

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.cfg = cfg
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dicts = task.source_dictionaries
        self.tgt_dicts = task.target_dictionaries
        self.channels = task.channels

        # optimize model for generation
        for model in self.models:
            model.prepare_for_inference_(cfg)

    def sample(
        self,
        sentences: List[Dict[str, str]],
        beam: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> List[str]:
        if isinstance(sentences, dict):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]

    def score(self, sentences: List[Dict[str, str]], **kwargs):
        raise NotImplementedError(
            "MultichannelGeneratorHubInterface doesn't support score() method"
        )

    def generate(
        self,
        tokenized_sentences: List[Dict[str, torch.LongTensor]],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if isinstance(tokenized_sentences, dict):
            return self.generate(
                [tokenized_sentences], beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in tqdm(
            self._build_batches(tokenized_sentences, skip_invalid_size_inputs)
        ):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                # The output of the generator is supposed to be a tensor of size (bsz x max_len x n_channels)
                # So we need to convert it to dictionary form
                for i in range(len(hypos)):
                    hypos[i]["tokens"] = {
                        channel: hypos[i]["tokens"][..., j]
                        for j, channel in enumerate(self.channels)
                    }
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = {
                    channel: self.string(source_tokens[channel], channel)
                    for channel in source_tokens
                }
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    # hypo["positional_scores"]: T x n_channels
                    pos_scores = {}
                    for c, channel in enumerate(source_tokens):
                        pos_scores[channel] = " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                hypo["positional_scores"][:, c].tolist(),
                            )
                        )
                    logger.info("P\t{}".format(pos_scores))

        return outputs

    def encode(self, sentence: Dict[str, str]) -> Dict[str, torch.LongTensor]:
        assert isinstance(
            sentence, dict
        ), "Input sentence is expected to be a dictionary over channels"
        assert set(sentence.keys()) == set(
            self.channels
        ), "Mismatch between input sentence keys and model channels ({} vs {})".format(
            set(sentence.keys()), set(self.channels)
        )
        encoded_sentence = {}
        for channel in sentence:
            sentence_channel = sentence[channel]
            sentence_channel = self.tokenize(sentence_channel)
            sentence_channel = self.apply_bpe(sentence_channel)
            sentence_channel = self.binarize(sentence_channel, channel)
            encoded_sentence[channel] = sentence_channel
        sentence_size = encoded_sentence[self.channels[0]].size()
        assert all(
            encoded_sentence[channel].size() == sentence_size
            for channel in encoded_sentence
        ), "Input tensors are expected to have the same size in all channels"
        return encoded_sentence

    def decode(self, tokens: Dict[str, torch.LongTensor]) -> Dict[str, str]:
        assert isinstance(
            tokens, dict
        ), "Input tokens are expected to be a dictionary over channels"
        assert set(tokens.keys()) == set(
            self.channels
        ), "Mismatch between input tokens keys and model channels ({} vs {})".format(
            set(tokens.keys()), set(self.channels)
        )
        decoded_sentence = {}
        for channel in tokens:
            tokens_channel = tokens[channel]
            sentence_channel = self.string(tokens_channel, channel)
            sentence_channel = self.remove_bpe(sentence_channel)
            sentence_channel = self.detokenize(sentence_channel)
            decoded_sentence[channel] = sentence_channel
        return decoded_sentence

    def binarize(self, sentence: str, channel: str) -> torch.LongTensor:
        return (
            self.src_dicts[channel].encode_line(sentence, add_if_not_exist=False).long()
        )

    def string(self, tokens: torch.LongTensor, channel: str) -> str:
        return self.tgt_dicts[channel].string(tokens)

    def _build_batches(
        self, tokens: List[Dict[str, List[int]]], skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([next(iter(d.values())).numel() for d in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator
