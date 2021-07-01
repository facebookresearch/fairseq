# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Dict

import torch

from fairseq.file_chunker_utils import Chunker
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
    ) -> Dict[str, int]:
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with Chunker(
            PathManager.get_local_path(filename), offset, end
        ) as line_iterator:
            for line in line_iterator:
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(
        filename, alignment_parser, consumer, offset=0, end=-1
    ) -> Dict[str, int]:
        nseq = 0

        with Chunker(
            PathManager.get_local_path(filename), offset, end
        ) as line_iterator:
            for line in line_iterator:
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
        return {"nseq": nseq}
