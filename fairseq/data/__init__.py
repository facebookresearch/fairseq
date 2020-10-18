# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset, FairseqIterableDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .add_target_dataset import AddTargetDataset
from .append_token_dataset import AppendTokenDataset
from .audio.raw_audio_dataset import FileAudioDataset
from .backtranslation_dataset import BacktranslationDataset
from .bucket_pad_length_dataset import BucketPadLengthDataset
from .colorize_dataset import ColorizeDataset
from .concat_dataset import ConcatDataset
from .concat_sentences_dataset import ConcatSentencesDataset
from .denoising_dataset import DenoisingDataset
from .id_dataset import IdDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .list_dataset import ListDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .lru_cache_dataset import LRUCacheDataset
from .mask_tokens_dataset import MaskTokensDataset
from .monolingual_dataset import MonolingualDataset
from .multi_corpus_sampled_dataset import MultiCorpusSampledDataset
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
from .resampling_dataset import ResamplingDataset
from .roll_dataset import RollDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .sort_dataset import SortDataset
from .strip_token_dataset import StripTokenDataset
from .subsample_dataset import SubsampleDataset
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .shorten_dataset import TruncateDataset, RandomCropDataset
from .multilingual.sampled_multi_dataset import SampledMultiDataset
from .multilingual.sampled_multi_epoch_dataset import SampledMultiEpochDataset
from .fasta_dataset import FastaDataset, EncodedFastaDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'AddTargetDataset',
    'AppendTokenDataset',
    'BacktranslationDataset',
    'BaseWrapperDataset',
    'BucketPadLengthDataset',
    'ColorizeDataset',
    'ConcatDataset',
    'ConcatSentencesDataset',
    'CountingIterator',
    'DenoisingDataset',
    'Dictionary',
    'EncodedFastaDataset',
    'EpochBatchIterator',
    'FairseqDataset',
    'FairseqIterableDataset',
    'FastaDataset',
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
    'MultiCorpusSampledDataset',
    'NestedDictionaryDataset',
    'NoisingDataset',
    'NumelDataset',
    'NumSamplesDataset',
    'OffsetTokensDataset',
    'PadDataset',
    'PrependDataset',
    'PrependTokenDataset',
    'ReplaceDataset',
    'RollDataset',
    'FileAudioDataset',
    'RawLabelDataset',
    'ResamplingDataset',
    'RightPadDataset',
    'RoundRobinZipDatasets',
    'SampledMultiDataset',
    'SampledMultiEpochDataset',
    'ShardedIterator',
    'SortDataset',
    'StripTokenDataset',
    'SubsampleDataset',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    'TruncateDataset',
    'TruncatedDictionary',
]
