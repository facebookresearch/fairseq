# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
from examples.textless_nlp.gslm.unit2speech.tacotron2.text import (
    EOS_TOK,
    SOS_TOK,
    code_to_sequence,
    text_to_sequence,
)
from examples.textless_nlp.gslm.unit2speech.tacotron2.utils import (
    load_code_dict,
)


class TacotronInputDataset:
    def __init__(self, hparams, append_str=""):
        self.is_text = getattr(hparams, "text_or_code", "text") == "text"
        if not self.is_text:
            self.code_dict = load_code_dict(
                hparams.code_dict, hparams.add_sos, hparams.add_eos
            )
            self.code_key = hparams.code_key
        self.add_sos = hparams.add_sos
        self.add_eos = hparams.add_eos
        self.collapse_code = hparams.collapse_code
        self.append_str = append_str

    def process_code(self, inp_str):
        inp_toks = inp_str.split()
        if self.add_sos:
            inp_toks = [SOS_TOK] + inp_toks
        if self.add_eos:
            inp_toks = inp_toks + [EOS_TOK]
        return code_to_sequence(inp_toks, self.code_dict, self.collapse_code)

    def process_text(self, inp_str):
        return text_to_sequence(inp_str, ["english_cleaners"])

    def get_tensor(self, inp_str):
        # uid, txt, inp_str = self._get_data(idx)
        inp_str = inp_str + self.append_str
        if self.is_text:
            inp_toks = self.process_text(inp_str)
        else:
            inp_toks = self.process_code(inp_str)
        return torch.from_numpy(np.array(inp_toks)).long()

    def __len__(self):
        return len(self.data)
