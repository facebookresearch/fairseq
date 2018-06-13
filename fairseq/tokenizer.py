# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re
import multiprocessing
import pickle
import tempfile

import torch

from fairseq import dictionary


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer:

    @staticmethod
    def build_dictionary(filename, tokenize=tokenize_line):
        dict = dictionary.Dictionary()
        Tokenizer.add_file_to_dictionary(filename, dict, tokenize)
        dict.finalize()
        return dict

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, worker_cnt=1, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False):
        if worker_cnt == 1:
            return Tokenizer.binarize_sequential(filename, dict, consumer, tokenize, append_eos, reverse_order)
        else:
            return Tokenizer.binarize_parallel(filename, dict, consumer, worker_cnt, tokenize, append_eos, reverse_order)

    @staticmethod
    def binarize_sequential(filename, dict, consumer, tokenize=tokenize_line,
            append_eos=True, reverse_order=False):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}


    @staticmethod
    def binarize_parallel(filename, dict, consumer, worker_cnt, tokenize=tokenize_line,
            append_eos=True, reverse_order=False):

        def binarize_worker(worker_id, tempfile):
            replaced = Counter()

            def replaced_consumer(word, idx):
                if idx == dict.unk_index and word != dict.unk_word:
                    replaced.update([word])

            ids_list = []
            nseq, ntok = 0, 0
            with open(filename, 'r') as f:
                for line_idx, line in enumerate(f):
                    if line_idx % worker_cnt == worker_id:
                        ids = Tokenizer.tokenize(
                            line=line,
                            dict=dict,
                            tokenize=tokenize,
                            add_if_not_exist=False,
                            consumer=replaced_consumer,
                            append_eos=append_eos,
                            reverse_order=reverse_order,
                        )
                        nseq += 1
                        ntok += len(ids)
                        ids_list.append(ids)

            ret = {'nseq': nseq, 'ntok': ntok, 'replaced': replaced, 'ids': ids_list}
            with open(tempfile, 'wb') as f:
                pickle.dump(ret, f)

        with tempfile.TemporaryDirectory() as temp_folder:
            temp_files = ['%s/%d' % (temp_folder, i) for i in range(worker_cnt)]
            thread_pool = [multiprocessing.Process(target=binarize_worker, args=(i, temp_files[i])) for i in range(worker_cnt)]
            for t in thread_pool:
                t.start()
            for t in thread_pool:
                t.join()

            worker_result = [pickle.load(open(temp_files[i], 'rb')) for i in range(worker_cnt)]

        nseq, ntok = 0, 0
        replaced = Counter()
        for r in worker_result:
            nseq += r['nseq']
            ntok += r['ntok']
            replaced.update(r['replaced'])

        for i in range(len(worker_result[0]['ids'])):
            for r in worker_result:
                if i < len(r['ids']):
                    consumer(r['ids'][i])

        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}


    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids
