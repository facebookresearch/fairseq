# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from fairseq.models.fairseq_model import FairseqLanguageModel


TAGS = {'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'}


class ClassBasedLanguageModel(FairseqLanguageModel):
    def __init__(self, lm, lm_generic, dictionary, tags):
        super().__init__(lm.decoder)
        self.generic_decoder = lm_generic.decoder
        self.dictionary = dictionary
        self.initial_tags_idxs = set()
        self.end_tags_idxs = set()
        for tag in tags:
            init_tag_idx = dictionary.index("<{}>".format(tag))
            end_tag_idx = dictionary.index("</{}>".format(tag))
            assert init_tag_idx != dictionary.unk_index, "<{}> was not found in the tgt dict".format(tag)
            assert end_tag_idx != dictionary.unk_index, "</{}> was not found in the tgt dict".format(tag)
            self.initial_tags_idxs.add(init_tag_idx)
            self.end_tags_idxs.add(end_tag_idx)

    def forward(self, src_tokens, **kwargs):
        out_scores = []
        for sample in src_tokens:
            inside_class_tag = False
            for i in range(sample.shape[0]):
                idx = -1 - i
                token = sample[idx].item()
                if token in self.initial_tags_idxs:
                    inside_class_tag = True
                    break
                if token in self.end_tags_idxs:
                    break
            if inside_class_tag:
                out_scores.append(self.decoder(sample[idx:].unsqueeze(0))[0][:, -1:, :])
            else:
                out_scores.append(self.generic_decoder(sample.unsqueeze(0))[0][:, -1:, :])
        return (torch.cat(out_scores), )
