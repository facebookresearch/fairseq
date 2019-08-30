# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .audio.raw_audio_dataset import FileAudioDataset
from .backtranslation_dataset import BacktranslationDataset
from .concat_dataset import ConcatDataset
from .concat_sentences_dataset import ConcatSentencesDataset
from .id_dataset import IdDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .list_dataset import ListDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .lru_cache_dataset import LRUCacheDataset
from .mask_tokens_dataset import MaskTokensDataset
from .monolingual_dataset import MonolingualDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .noising import NoisingDataset
from .numel_dataset import NumelDataset
from .num_samples_dataset import NumSamplesDataset
from .offset_tokens_dataset import OffsetTokensDataset
from .pad_dataset import LeftPadDataset, PadDataset, RightPadDataset
from .prepend_dataset import PrependDataset
from .prepend_token_dataset import PrependTokenDataset
from .raw_label_dataset import RawLabelDataset
from .replace_dataset import ReplaceDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .sharded_dataset import ShardedDataset
from .sort_dataset import SortDataset
from .strip_token_dataset import StripTokenDataset
from .subsample_dataset import SubsampleDataset
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .truncate_dataset import TruncateDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'BacktranslationDataset',
    'BaseWrapperDataset',
    'ConcatDataset',
    'ConcatSentencesDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IdDataset',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'LeftPadDataset',
    'ListDataset',
    'LMContextWindowDataset',
    'LRUCacheDataset',
    'MaskTokensDataset',
    'MMapIndexedDataset',
    'MonolingualDataset',
    'NestedDictionaryDataset',
    'NoisingDataset',
    'NumelDataset',
    'NumSamplesDataset',
    "OffsetTokensDataset",
    'PadDataset',
    'PrependDataset',
    'PrependTokenDataset',
    'ReplaceDataset',
    'FileAudioDataset',
    "RawLabelDataset",
    'RightPadDataset',
    'RoundRobinZipDatasets',
    'ShardedDataset',
    'ShardedIterator',
    'SortDataset',
    'StripTokenDataset',
    'SubsampleDataset',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    "TruncateDataset",
    'TruncatedDictionary',
]
