# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . scorer import SimulScorer
from . import register_scorer


@register_scorer("text")
class SimulTextScorer(SimulScorer):
    def __init__(self, args):
        super().__init__(args)
        self.data = {
            "src": self._load_text_file(args.src_file, split=True),
            "tgt": self._load_text_file(args.tgt_file, split=False)
        }

    def send_src(self, sent_id, *args):
        if self.steps[sent_id] >= len(self.data["src"][sent_id]):
            dict_to_return = {
                "sent_id": sent_id,
                "segment_id": self.steps[sent_id],
                "segment": self.eos
            }
            # Consider EOS
            self.steps[sent_id] = len(self.data["src"][sent_id]) + 1
        else:
            dict_to_return = {
                "sent_id": sent_id,
                "segment_id": self.steps[sent_id],
                "segment": self.data["src"][sent_id][self.steps[sent_id]]
            }

            self.steps[sent_id] += 1

        return dict_to_return

    def src_lengths(self):
        # +1 for eos
        return [len(sent) + 1 for sent in self.data["src"]]
