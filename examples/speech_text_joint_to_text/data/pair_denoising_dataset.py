# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import re

import torch

from fairseq.data import data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset


# Part of the code is modified from DenoisingDataset
# compared with DenoisingDataset, no permute_sentences or documents (rotate_ratio, permute_sentence_ratio)
class LanguagePairDenoisingDataset(LanguagePairDataset):
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt,
        tgt_sizes,
        tgt_dict,
        mask_idx,
        mask_whole_words,
        seed,
        args,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        super().__init__(
            src,
            src_sizes,
            src_dict,
            tgt,
            tgt_sizes,
            tgt_dict,
            left_pad_source,
            left_pad_target,
            shuffle,
            input_feeding,
            remove_eos_from_source,
            append_eos_to_target,
            align_dataset,
            constraints,
            append_bos,
            eos,
            num_buckets,
            src_lang_id,
            tgt_lang_id,
            pad_to_multiple,
        )

        self.mask_idx = mask_idx
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = args.mask
        self.random_ratio = args.mask_random
        self.insert_ratio = args.insert

        self.replace_length = args.replace_length

        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if args.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={args.mask_length}")
        if args.mask_length == "subword" and args.replace_length not in [0, 1]:
            raise ValueError("if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if args.mask_length == "span-poisson":
            # Text infilling: "A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3). Each span is replaced with a single [MASK] token. 0-length spans correspond to the insertion of [MASK] tokens."
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0
        self.seed = seed

        def _is_phoneme(x):
            if re.search("<lang:", x) or x in (
                "<mask>",
                "<sil>",
                "<pad>",
                "<s>",
                "</s>",
                "<unk>",
            ):
                return False
            return True

        self.voc_valid_ids = torch.LongTensor(
            [i for i, x in enumerate(self.src_dict.symbols) if _is_phoneme(x)]
        )
        self.voc_valid_size = self.voc_valid_ids.size(0)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = copy.deepcopy(self.src[index])
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            source = src_item
            assert source[-1] == self.eos
            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)
            src_item = source

        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if src_item[0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), src_item])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if src_item[-1] == eos:
                src_item = src_item[:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        if self.src_lang_id is not None:
            example["src_lang_id"] = self.src_lang_id
        if self.tgt_lang_id is not None:
            example["tgt_lang_id"] = self.tgt_lang_id
        return example

    # following functions are borrowed from denoising_dataset
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = self.voc_valid_ids[
                torch.randint(0, self.voc_valid_size - 1, size=(mask_random.sum(),))
            ]

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = self.voc_valid_ids[
                        torch.randint(
                            0, self.voc_valid_size - 1, size=(mask_random.sum(),)
                        )
                    ]
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = self.voc_valid_ids[
                        torch.randint(
                            0, self.voc_valid_size - 1, size=(mask_random.sum(),)
                        )
                    ]

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = self.voc_valid_ids[
            torch.randint(0, self.voc_valid_size - 1, size=(num_random,))
        ]

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result
