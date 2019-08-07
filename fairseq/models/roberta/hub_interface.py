# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data import encoders


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        for s in addl_sentences:
            bpe_sentence += ' </s> ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor):
        features = self.extract_features(tokens)
        logits = self.model.classification_heads[head](features)
        return F.log_softmax(logits, dim=-1)

    def extract_features_aligned_to_words(self, sentence: str, return_all_hiddens: bool = False) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        from fairseq.models.roberta import alignment_utils
        from spacy.tokens import Doc

        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()

        # tokenize both with GPT-2 BPE and spaCy
        bpe_toks = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

        # extract features and align them
        features = self.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(self, features, alignment)

        # wrap in spaCy Doc
        doc = Doc(
            nlp.vocab,
            words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'],
            spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False],
        )
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        return doc
