from fairseq.data.strip_token_dataset import StripTokenDataset
from fairseq.data.shorten_dataset import TruncateDataset
import os
from fairseq.data import indexed_dataset
from fairseq.data.append_token_dataset import AppendTokenDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data.fairseq_dataset import FairseqDataset
import numpy as np
import torch
from fairseq.data import data_utils
from fairseq.data.noising import UnsupervisedMTNoising, NoisingDataset
from fairseq.data.pad_dataset import PadDataset
from fairseq.data.numel_dataset import NumelDataset
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.prepend_token_dataset import PrependTokenDataset
from fairseq.data.denoising_dataset import DenoisingDataset
from fairseq.data import plasma_utils
from fairseq.data.indexed_dataset import best_fitting_int_dtype
from typing import Tuple
import itertools
import logging

logger = logging.getLogger(__name__)


def fairseq_filter_indices_by_size(dataset, indices, max_sizes):
    """
    Filter a list of sample indices. Remove those that are longer than
    specified in *max_sizes*.

    WARNING: don't update, override method in child classes

    Args:
        indices (np.array): original array of sample indices
        max_sizes (int or list[int] or tuple[int]): max sample size,
            can be defined separately for src and tgt (then list or tuple)

    Returns:
        np.array: filtered sample array
        list: list of removed indices
    """
    if isinstance(max_sizes, float) or isinstance(max_sizes, int):
        if hasattr(dataset, "sizes") and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_sizes].tolist()
            indices = indices[dataset.sizes[indices] <= max_sizes]
        elif (
            hasattr(dataset, "sizes")
            and isinstance(dataset.sizes, list)
            and len(dataset.sizes) == 1
        ):
            ignored = indices[dataset.sizes[0][indices] > max_sizes].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_sizes]
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, dataset.size, max_sizes
            )
    else:
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, dataset.size, max_sizes
        )
    return indices, ignored


class SwavMultilingualMatchDataset(ConcatDataset):
    """
    This will randomly pick a subdataset pivot,
        for each sample from the pivot, sample 1 samples from the rest of the
        subdatasets
        e.g: idx -> dataset_idx, sample_idx
            r_sample_idx for each r_dataset_idx in range(len(datasets)) if r_dataset_idx != dataset_idx
            concat(sample_idx + r_sample_idx)
        # this will multiple the number of samples in a batch,
            consider reduce the batch size
    """
    def __init__(self, datasets, max_positions=None, sample_ratios=1, ignore_invalid_inputs=False):
        # datasets may contain nested dict
        super(SwavMultilingualMatchDataset, self).__init__(datasets, sample_ratios)
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        # build available indices for datasets
        self.avail_indices = [
            self._build_available_indices(d, max_positions, ignore_invalid_inputs)
            for d in datasets
        ]

    def _filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        copy from fairseq_task
        """
        indices, ignored = fairseq_filter_indices_by_size(dataset, indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

    def _build_available_indices(self, dataset, max_positions=None, ignore_invalid_inputs=False):
        indices = np.arange(len(dataset))
        assert isinstance(dataset, FairseqDataset), f'type dataset{type(dataset)}'
        indices = self._filter_indices_by_size(
            indices, dataset, max_positions=max_positions, ignore_invalid_inputs=ignore_invalid_inputs)
        return indices

    def __getitem__(self, idx):
        p_dataset_idx, p_sample_idx = self._get_dataset_and_sample_index(idx)
        samples = [self.datasets[p_dataset_idx][p_sample_idx]]
        for d_idx in np.delete(np.arange(len(self.datasets)), p_dataset_idx):
            indices = self.avail_indices[d_idx]
            samples.append(self.datasets[d_idx][
                indices[np.random.randint(0, len(indices))]])
        return samples

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        concat_samples = list(itertools.chain.from_iterable(samples))
        return super().collater(concat_samples, **extra_args)


class SwavExtrapolateNoisingDataset(NoisingDataset):
    """
    Extrapolate the dataset by noising the sentence rand_factor times
    e.g:
        from one item with __getitem__
        noise the items rand_factor times and return the list of items
        make sure each samples in items are distinctly noised
    """
    def __init__(
        self,
        src_dataset,
        src_dict,
        seed,
        rand_factor,
        noiser=None,
        noising_class=UnsupervisedMTNoising,
        nonoise_duplicate=False,
        **kwargs
    ):
        super().__init__(src_dataset, src_dict, seed, noiser, noising_class, **kwargs)
        self.rand_factor = rand_factor
        self.nonoise_duplicate = nonoise_duplicate

    def _noise_tokens(self, tokens, lengths, index, rand_idx):
        # Transpose src tokens to fit expected shape of x in noising function
        # (batch size, sequence length) -> (sequence length, batch size)
        src_tokens_t = torch.t(tokens)

        with data_utils.numpy_seed((self.seed + len(self) * rand_idx + index) % (2 ** 30)):
            noisy_src_tokens = self.noiser.noising(src_tokens_t, lengths)

        # Transpose back to expected src_tokens format
        # (sequence length, 1) -> (1, sequence length)
        noisy_src_tokens = torch.t(noisy_src_tokens)
        return noisy_src_tokens[0]

    def __getitem__(self, index):
        """
        Returns a single noisy sample. Multiple samples are fed to the collater
        create a noising dataset batch.
        """
        if self.nonoise_duplicate:
            return [self.src_dataset[index]] * self.rand_factor
            
        src_tokens = self.src_dataset[index]
        src_lengths = torch.LongTensor([len(src_tokens)])
        src_tokens = src_tokens.unsqueeze(0)

        tokens_list = []
        for rand_idx in range(self.rand_factor):
            toks = self._noise_tokens(src_tokens, src_lengths, index, rand_idx)
            tokens_list.append(toks)
        return tokens_list

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)


def swav_extrapolate_collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a LIST of list of 1d tensors into a padded 2d tensor.
    order:
    out = prot_out[bsz * crop_id: bsz * (crop_id + 1)]
    Order:
        number is index in batch, abc is crops of an item: 
        [a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4]
    """
    rand_factor = len(values[0])
    bsz = len(values)
    values_t = [[values[j][i] for j in range(bsz)] for i in range(rand_factor)]
    values_f = list(itertools.chain.from_iterable(values_t))

    size = max(v.size(0) for v in values_f)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values_f) if pad_to_bsz is None else max(len(values_f), pad_to_bsz)
    res = values_f[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values_f):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
    return res


class SwavExtrapolatePrependTokenDataset(PrependTokenDataset):
    def __init__(self, dataset, token=None):
        # assert isinstance(dataset, SwavExtrapolateNoisingDataset)
        super().__init__(dataset)
        self.token = token
        if token is not None:
            self._sizes = dataset.sizes + 1
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            assert isinstance(item, list)
            item = [torch.cat([x.new([self.token]), x]) for x in item]
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        # NOTE(nxphi) num_tokens take the 1st from list
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        # NOTE(nxphi) size take the 1st from list
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n


class SwavExtrapolatePadDataset(PadDataset):
    def collater(self, samples):
        return swav_extrapolate_collate_tokens(samples, self.pad_idx, left_pad=self.left_pad)
        

class SwavExtrapolateNumelDataset(NumelDataset):
    def __getitem__(self, index):
        item = self.dataset[index]
        # assert isinstance(item, list)
        if torch.is_tensor(item[0]):
            return [torch.numel(x) for x in item]
        else:
            raise NotImplementedError('not impl yet')

    def collater(self, samples):
        if self.reduce:
            return sum(sum(x) for x in samples)
        else:
            # return torch.tensor(samples)
            rand_factor = len(samples[0])
            bsz = len(samples)
            lengths = [[samples[j][i] for j in range(bsz)] for i in range(rand_factor)]
            lengths_f = list(itertools.chain.from_iterable(lengths))
            return torch.tensor(lengths_f)


class LangIdDataset(NumelDataset):
    def __init__(self, dataset, lang_id, reduce):
        super().__init__(dataset, reduce=reduce)
        self.lang_id = lang_id

    def __getitem__(self, index):
        return self.lang_id

    def collater(self, samples):
        if self.reduce:
            # return sum(samples)
            raise ValueError
        else:
            return torch.tensor(samples)


class SwavExtrapolateLangIdDataset(SwavExtrapolateNumelDataset):
    def __init__(self, dataset, lang_id, reduce):
        super().__init__(dataset, reduce=reduce)
        self.lang_id = lang_id

    def __getitem__(self, index):
        item = self.dataset[index]
        if torch.is_tensor(item[0]):
            return [self.lang_id] * len(item)
        else:
            raise NotImplementedError('not impl yet')

    def collater(self, samples):
        if self.reduce:
            raise ValueError
        else:
            rand_factor = len(samples[0])
            bsz = len(samples)
            values_t = [[samples[j][i] for j in range(bsz)] for i in range(rand_factor)]
            langs = list(itertools.chain.from_iterable(values_t))
            return torch.tensor(langs)


def swav_extra_denoise_collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    sort=False,
    get_swav_input=True,
):
    assert left_pad_source is False, f'left_pad_source true not support'
    assert (get_swav_input and not sort) or (not get_swav_input and sort)
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # lang_ids
    lang_ids = torch.LongTensor([s["lang_id"] for s in samples])
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])

    sort_order = None
    if sort:
        # sort by descending source length
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        lang_ids = lang_ids.index_select(0, sort_order)

    # build swav inputs (nosort)
    net_swav_input = None
    if get_swav_input:
        assert not sort, f'sort for swav input not supported yet'
        swav_sources = [s["swav_sources"] for s in samples]
        swav_langs = [s["swav_langs"] for s in samples]
        rand_factor = len(swav_sources[0])
        bsz = len(swav_sources)
        swav_sources_t = [[swav_sources[j][i] for j in range(bsz)] for i in range(rand_factor)]
        swav_langs_t = [[swav_langs[j][i] for j in range(bsz)] for i in range(rand_factor)]
        swav_src_tokens = swav_extrapolate_collate_tokens(
            swav_sources, pad_idx, eos_idx=None, left_pad=left_pad_source, pad_to_length=pad_to_length)

        swav_langs = torch.LongTensor(list(itertools.chain.from_iterable(swav_langs_t)))
        swav_src_lengths = torch.LongTensor([
            s.numel()
            for s in list(itertools.chain.from_iterable(swav_sources_t))]
        )
        assert (
            swav_src_tokens.size(0) == swav_src_lengths.size(0) == swav_langs.size(0)
        ), f'{swav_src_tokens.size()=} != {swav_src_lengths.size()=} != {swav_langs.size()}'
        net_swav_input = {
            "src_tokens": swav_src_tokens,
            "src_lengths": swav_src_lengths,
            "src_langs": swav_langs,
        }
    
    # build target
    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        if sort:
            target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            if sort:
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "lang_id": lang_ids,
        "target": target,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }
    if get_swav_input:
        batch['net_swav_input'] = net_swav_input
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class SwavExtrapolateDenoisingDataset(DenoisingDataset):
    def __init__(
        self,
        lang_id,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        rand_factor,
        eos=None,
        item_transform_func=None,
    ):
        super().__init__(
            dataset, sizes, vocab, mask_idx, mask_whole_words, 
            shuffle, seed, args, eos=eos, item_transform_func=item_transform_func
        )
        self.lang_id = lang_id
        self.rand_factor = rand_factor

    def noise(self, source):
        if self.permute_sentence_ratio > 0.0:
            source = self.permute_sentences(source, self.permute_sentence_ratio)

        if self.mask_ratio > 0:
            source = self.add_whole_word_mask(source, self.mask_ratio)

        if self.insert_ratio > 0:
            source = self.add_insertion_noise(source, self.insert_ratio)

        if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
            source = self.add_rolling_noise(source)
        return source

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()
            source = self.noise(source)
            
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)
        
        swav_sources = [source.clone()]
        for rand_idx in range(1, self.rand_factor):
            with data_utils.numpy_seed(self.seed, self.epoch, index, rand_idx):
                swav_sources.append(self.noise(source))

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[-1] == self.eos
        return {
            "id": index,
            "source": source,
            "target": target,
            "lang_id": self.lang_id,
            "swav_sources": swav_sources,
            "swav_langs": [self.lang_id] * self.rand_factor
        }

    def collater(self, samples, pad_to_length=None):
        return swav_extra_denoise_collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length,
            sort=False, get_swav_input=True,
        )


def get_sent2doc_slice_indices_slow(sizes, break_mode, block_size, document_sep_len):
    # NOTE: break_mode and document_sep_len are not used for now
    assert break_mode == 'complete'
    cumsum = np.cumsum(sizes)
    ptr = 0
    slice_indices = []
    for i in range(0, len(sizes)):
        if i == 0:
            # special case
            while cumsum[ptr] <= block_size:
                if cumsum[ptr + 1] > block_size:
                    break
                ptr += 1
            slice_indices.append([0, cumsum[ptr]])
        else:
            # normal case
            while ptr + 1 < sizes.shape[0] and cumsum[ptr] - cumsum[i - 1] <= block_size:
                if cumsum[ptr + 1] - cumsum[i - 1] > block_size:
                    break
                else:
                    ptr += 1
            slice_indices.append([cumsum[i - 1], cumsum[ptr]])
    slice_indices = np.array(slice_indices)
    return slice_indices


class Sent2DocTokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks, but always starting at each sentences
    Different from TokenBlockDataset is that each sample reference to each sentence
        in the dataset, instead of the next sentence after finishing previous block

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            FIXME nxphi: the only supported mode is complete!
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False). FIXME nxphi: currently only support False
        document_sep_len (int, optional): FIXME nxphi: this may not be used at all.
            document separator size (required for 'complete_doc' break mode). 
            Typically 1 if the sentences have eos and 0 otherwise.
    """

    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        use_plasma_view=False,
        split_path=None,
        plasma_path=None,
    ):

        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets
        if break_mode == "eos" or block_size is None:
            raise ValueError(f'not support break_mode {break_mode}, consider not using this Class')

        assert len(dataset) > 0

        assert len(dataset) == len(sizes)
        _sizes, block_to_dataset_index, slice_indices = self._build_slice_indices(
            sizes, break_mode, document_sep_len, block_size
        )
        if use_plasma_view:
            plasma_id = (block_size, document_sep_len, str(break_mode), len(dataset))
            self._slice_indices = plasma_utils.PlasmaView(
                slice_indices, split_path, (plasma_id, 0), plasma_path=plasma_path
            )
            self._sizes = plasma_utils.PlasmaView(
                _sizes, split_path, (plasma_id, 1), plasma_path=plasma_path
            )
            self._block_to_dataset_index = plasma_utils.PlasmaView(
                block_to_dataset_index, split_path, (plasma_id, 2), plasma_path=plasma_path,
            )
        else:
            self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
            self._sizes = plasma_utils.PlasmaArray(_sizes)
            self._block_to_dataset_index = plasma_utils.PlasmaArray(
                block_to_dataset_index
            )

    @staticmethod
    def _build_slice_indices(
        sizes, break_mode, document_sep_len, block_size
    ) -> Tuple[np.ndarray]:
        """Use token_block_utils_fast to build arrays for indexing into self.dataset"""
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_block_to_dataset_index_fast
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else "none"

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" or block_size is None:
            # block_size = 0
            raise ValueError(f'not support break_mode {break_mode}, consider not using this Class')

        slice_indices = get_sent2doc_slice_indices_slow(
            sizes, str(break_mode), block_size, document_sep_len
        )
        _sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            # block_to_dataset_index = np.stack(
            #     [
            #         np.arange(len(sizes)),  # starting index in dataset
            #         np.zeros(
            #             len(sizes), dtype=np.compat.long
            #         ),  # starting offset within starting index
            #         np.arange(len(sizes)),  # ending index in dataset
            #     ],
            #     1,
            # )
            raise ValueError(f'not support break_mode {break_mode}, consider not using this Class')
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes, slice_indices,
            )
        size_dtype = np.uint16 if block_size < 65535 else np.uint32
        num_tokens = slice_indices[-1].max()
        slice_indices_dtype = best_fitting_int_dtype(num_tokens)
        slice_indices = slice_indices.astype(slice_indices_dtype)
        _sizes = _sizes.astype(size_dtype)
        block_to_dataset_index = block_to_dataset_index.astype(slice_indices_dtype)
        return _sizes, block_to_dataset_index, slice_indices

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )
        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0:e - 1]])
                past_target = torch.cat(
                    [item.new([self.pad, self.eos]), buffer[0:e - 2]]
                )
            else:
                source = buffer[s - 1:e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0:e - 2]])
                else:
                    past_target = buffer[s - 2:e - 2]

            return source, item, past_target

        return item

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )


class Sent2ContinueSentsTokenBlockDataset(Sent2DocTokenBlockDataset):
    """
    Simimlar to Sent2DocTokenBlockDataset, but concatenate sentences into batch
        build net_input dict inside
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        use_plasma_view=False,
        split_path=None,
        plasma_path=None
    ):
        super().__init__(
            dataset, sizes, block_size, pad, eos, break_mode=break_mode,
            include_targets=include_targets, document_sep_len=document_sep_len, 
            use_plasma_view=use_plasma_view, split_path=split_path, plasma_path=plasma_path)
        """
        "src_tokens": PadDataset(
            lang_bias_dataset,
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        ),
        "src_lengths": NumelDataset(lang_bias_dataset, reduce=False),
        "src_langs": RawLabelDataset([lang_id] * lang_bias_dataset.sizes.shape[0]),
        """

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )
        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]
        eos_idxs = (item == self.eos).nonzero()[:, 0].tolist()
        eos_idxs = ([0] + eos_idxs)
        sents = [item[eos_idxs[i]:eos_idxs[i + 1]] for i in range(len(eos_idxs) - 1)]

        # ----
        if self.include_targets:
            raise NotImplementedError
        return sents


class SentInFixedDocDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, doc_len, max_len, seed, mode='rand') -> None:
        super().__init__()
        self.dataset = dataset
        self.sizes = dataset.sizes
        self.doc_len = doc_len
        self.mode = mode
        self.max_len = max_len
        self.seed = seed
        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch        

    def extract_mul_items(self, start_index):
        item_list = []
        for idx in range(start_index, start_index + self.doc_len):
            if idx >= 0 and idx < len(self.dataset):
                item_list.append(self.dataset[idx])
        item_list = [x for x in item_list if len(x) <= self.max_len]
        return item_list

    def infer_start_index(self, index):
        if self.mode == "first":
            start_index = index
        elif self.mode == "last":
            start_index = index - self.doc_len + 1
        elif self.mode == "rand":
            idx = np.random.randint(0, self.doc_len)
            start_index = index - idx
        else:
            raise ValueError(f'mode not found: {self.mode}')
        return start_index

    def __getitem__(self, index):
        assert len(self.dataset[index]) <= self.max_len
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            start_index = self.infer_start_index(index)
            item_list = self.extract_mul_items(start_index)
        return item_list

    def __len__(self):
        return len(self.dataset)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        if self.dataset.supports_prefetch:
            self.dataset.prefetch(indices)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)


class NoisingSentInFixedDocDataset(SentInFixedDocDataset):
    def __init__(
        self, dataset, doc_len, max_len, seed, src_dict, mode='rand', 
        noiser=None, noising_class=UnsupervisedMTNoising, **kwargs
    ):
        super().__init__(dataset, doc_len, max_len, seed, mode=mode)
        self.src_dict = src_dict
        self.noiser = (
            noiser
            if noiser is not None
            else noising_class(
                dictionary=self.src_dict,
                **kwargs,
            )
        )

    def _noise_tokens(self, tokens, lengths, index, rand_idx):
        # Transpose src tokens to fit expected shape of x in noising function
        # (batch size, sequence length) -> (sequence length, batch size)
        src_tokens_t = torch.t(tokens)

        # with data_utils.numpy_seed((self.seed + len(self) * rand_idx + index) % (2 ** 30)):
        with data_utils.numpy_seed(self.seed, self.epoch, rand_idx, index):
            
            noisy_src_tokens = self.noiser.noising(src_tokens_t, lengths)

        # Transpose back to expected src_tokens format
        # (sequence length, 1) -> (1, sequence length)
        noisy_src_tokens = torch.t(noisy_src_tokens)
        return noisy_src_tokens[0]

    def extract_mul_items(self, start_index):
        item_list = []
        for i, idx in enumerate(range(start_index, start_index + self.doc_len)):
            if idx >= 0 and idx < len(self.dataset):
                item = self.dataset[idx]
                src_tokens = item.unsqueeze(0)
                src_lengths = torch.LongTensor([len(src_tokens)])
                noised_item = self._noise_tokens(src_tokens, src_lengths, idx, i)
                item_list.append(noised_item)
        item_list = [x for x in item_list if len(x) <= self.max_len]
        return item_list


def list_concat_collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a LIST of list of 1d tensors into a padded 2d tensor.
    order:
    out = prot_out[bsz * crop_id: bsz * (crop_id + 1)]

    nums is batch, abc is crops: [a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4]
    """
    assert all(isinstance(x, (list, tuple)) for x in values)
    values_f = list(itertools.chain.from_iterable(values))

    size = max(v.size(0) for v in values_f)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values_f) if pad_to_bsz is None else max(len(values_f), pad_to_bsz)
    res = values_f[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values_f):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
    return res


class ListConcatNumelDataset(SwavExtrapolateNumelDataset):
    def __getitem__(self, index):
        item = self.dataset[index]
        assert isinstance(item, list)
        if torch.is_tensor(item[0]):
            return [torch.numel(x) for x in item]
        else:
            raise NotImplementedError('not impl yet')

    def collater(self, samples):
        if self.reduce:
            return sum(sum(x) for x in samples)
        else:
            assert all(isinstance(x, (list, tuple)) for x in samples)
            lengths = list(itertools.chain.from_iterable(samples))
            return torch.tensor(lengths)


class ListConcatLangIdDataset(SwavExtrapolateLangIdDataset):
    def collater(self, samples):
        if self.reduce:
            return sum(sum(x) for x in samples)
        else:
            assert all(isinstance(x, (list, tuple)) for x in samples)
            lengths = list(itertools.chain.from_iterable(samples))
            return torch.tensor(lengths)


class ListConcatPadDataset(PadDataset):
    def collater(self, samples):
        return list_concat_collate_tokens(samples, self.pad_idx, left_pad=self.left_pad)


class LangBiasAttachedSwavDataset(FairseqDataset):
    def __init__(
        self, dataset, lang_biases, 
        input_keys="net_swav_input", bias_key="lang_bias"
    ):
        super().__init__()
        self.dataset = dataset
        self.lang_biases = lang_biases
        self.sizes = dataset.sizes
        self.input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        self.bias_key = bias_key
        # self.lang_bias = plasma_utils.PlasmaArray(torch.tensor(
        self.lang_bias = torch.tensor(
            np.concatenate([x[None, :, :] for x in lang_biases]), dtype=torch.float32
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        batch = self.dataset.collater(samples)
        if not bool(batch):
            return batch
        for k in self.input_keys:
            if k in batch:
                batch[k][self.bias_key] = self.lang_bias
        return batch

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)


def collate_pair_score(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0:lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)
    
    if samples[0].get("weights", None) is not None:
        # Collate the weights:
        weights = torch.FloatTensor([x.get("weights") for x in samples])
        batch['weights'] = weights.index_select(0, sort_order)

    return batch


class LanguagePairWeightDataset(LanguagePairDataset):
    """
    Language Pair Dataset with weights values for each pair
        with key 'weights'
    """
    def __init__(
        self,
        weights,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
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
            src, src_sizes, src_dict, tgt=tgt, tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict, left_pad_source=left_pad_source, left_pad_target=left_pad_target, 
            shuffle=shuffle, input_feeding=input_feeding, remove_eos_from_source=remove_eos_from_source, 
            append_eos_to_target=append_eos_to_target, align_dataset=align_dataset, 
            constraints=constraints, append_bos=append_bos, eos=eos, num_buckets=num_buckets, 
            src_lang_id=src_lang_id, tgt_lang_id=tgt_lang_id, pad_to_multiple=pad_to_multiple
        )
        self.pair_weights = weights
        assert len(self.pair_weights) == len(self), f'{len(self.pair_weights)=} != {len(self)=}'

    def __getitem__(self, index):
        item = super().__getitem__(index)
        pair_score = self.pair_weights[index]
        item['weights'] = pair_score
        return item

    def collater(self, samples, pad_to_length=None):
        res = collate_pair_score(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
                res["net_input"]["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens.device)
                )
        return res


def load_langpair_weights_dataset(
    data_path,
    split,
    weights,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    no_append_lang_src=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        if not no_append_lang_src:
            src_dataset = AppendTokenDataset(
                src_dataset, src_dict.index("[{}]".format(src))
            )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    if weights is not None:
        return LanguagePairWeightDataset(
            weights,
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )
    else:
        return LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )