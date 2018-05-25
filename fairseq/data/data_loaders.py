# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from fairseq.data import LanguagePairDataset, MonolingualDataset
from fairseq.data.data_utils import infer_language_pair


def load_dataset(args, splits, is_raw):
    """ Detect if we have a multi language dataset, or a single language dataset """
    if args.source_lang is None and args.target_lang is None:
        # find language pair automatically
        args.source_lang, args.target_lang = infer_language_pair(args.data, splits)
    if args.source_lang is None and args.target_lang is None and all(
            os.path.exists(os.path.join(args.data, '{}.bin'.format(split))) for split in splits):
        cls = MonolingualDataset
    else:
        cls = LanguagePairDataset
    return cls.create_dataset(args, splits, is_raw)
