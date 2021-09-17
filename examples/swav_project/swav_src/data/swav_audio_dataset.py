
import itertools
import logging
import os
import re
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset
from fairseq.data.data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)

from fairseq.data.audio.raw_audio_dataset import RawAudioDataset, FileAudioDataset, BinarizedAudioDataset


logger = logging.getLogger(__name__)



class SwavExtrapolateRawAudioDataset(RawAudioDataset):
    def __init__(
        self,
        rand_factor,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(sample_rate, max_sample_size=max_sample_size, min_sample_size=min_sample_size, 
            shuffle=shuffle, pad=pad, normalize=normalize, compute_mask_indices=compute_mask_indices, **mask_compute_kwargs)
        self.rand_factor = rand_factor
    
    def __getitem__(self, index):
        # NOTE: expect to return a list
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)
    
    def _collate(self, sources, ids):
        sizes = [len(s) for s in sources]
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        out = {
            "id": torch.LongTensor(ids)
        }
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[idx] for idx in ids)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        # source is a list
        assert all(isinstance(s["source"], list) for s in samples)

        sources = [s["source"][0] for s in samples]
        sizes = [len(s) for s in sources]
        ids = [s["id"] for s in samples]

        # swav sources
        swav_sources = [s["source"] for s in samples]
        rand_factor = len(swav_sources[0])
        bsz = len(swav_sources)
        swav_sources_t = list(itertools.chain.from_iterable(
            [[swav_sources[j][i] for j in range(bsz)] for i in range(rand_factor)]))
        swav_ids = list(itertools.chain.from_iterable(
            [[samples[j]["id"] for j in range(bsz)] for i in range(rand_factor)]))

        batch = self._collate(sources, ids)
        swav_batch = self._collate(swav_sources_t, swav_ids)
        
        if "lang_id" in samples[0]:
            swav_langs = [s["lang_id"] for s in samples]
            swav_langs_t = [[swav_langs[j][i] for j in range(bsz)] for i in range(rand_factor)]
            swav_langs = torch.LongTensor(list(itertools.chain.from_iterable(swav_langs_t)))
            swav_batch['net_input']['src_langs'] = swav_langs
            # logger.warning(f'{swav_langs=}')
        batch['net_swav_input'] = swav_batch['net_input']
        return batch


class BinarizedSwavExtrapolateNoNoiseAudioDataset(SwavExtrapolateRawAudioDataset):
    def __init__(
        self,
        rand_factor,
        data_dir,
        split,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            rand_factor=rand_factor,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        from fairseq.data import data_utils, Dictionary

        self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

        root_path = os.path.join(data_dir, f"{split}.root")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                self.root_dir = next(f).strip()
        else:
            self.root_dir = None

        fnames_path = os.path.join(data_dir, split)
        self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
        lengths_path = os.path.join(data_dir, f"{split}.lengths")

        with open(lengths_path, "r") as f:
            for line in f:
                sz = int(line.rstrip())
                assert (
                    sz >= min_sample_size
                ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
                self.sizes.append(sz)

        self.sizes = np.array(self.sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)
        logger.info(f"loaded {len(self.fnames)} samples")
    
    def __getitem__(self, index):
        import soundfile as sf

        fname = self.fnames_dict.string(self.fnames[index], separator="")
        if self.root_dir:
            fname = os.path.join(self.root_dir, fname)

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)

        return {
            "id": index, 
            "source": [feats] * self.rand_factor
        }



class FileSwavExtrapolateNoNoiseAudioDataset(SwavExtrapolateRawAudioDataset):
    def __init__(
        self,
        rand_factor,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            rand_factor=rand_factor,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        # with open(manifest_path, "r") as f:
        #     self.root_dir = f.readline().strip()
        #     for i, line in enumerate(f):
        #         items = line.strip().split("\t")
        #         assert len(items) == 2, line
        #         sz = int(items[1])
        #         if min_sample_size is not None and sz < min_sample_size:
        #             skipped += 1
        #             self.skipped_indices.add(i)
        #             continue
        #         self.fnames.append(items[0])
        #         sizes.append(sz)

        self.root_dirs = []
        with open(manifest_path, "r") as f:
            root_dir = None
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                if len(items) == 1:
                    root_dir = items[0].strip()
                    assert os.path.exists(root_dir)
                    self.root_dirs.append(root_dir)
                else:
                    sz = int(items[1])
                    if min_sample_size is not None and sz < min_sample_size:
                        skipped += 1
                        self.skipped_indices.add(i)
                        continue
                    path = os.path.join(root_dir, str(items[0]))
                    self.fnames.append(path)
                    sizes.append(sz)
                
        logger.info(f"loaded {len(self.fnames)}, root_dirs {len(self.root_dirs)} skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        import soundfile as sf

        # path_or_fp = os.path.join(self.root_dir, str(self.fnames[index]))
        path_or_fp = str(self.fnames[index])
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {
            "id": index, 
            "source": [feats] * self.rand_factor
        }


class FileSwavExtrapolateNoNoiseLangAudioDataset(FileSwavExtrapolateNoNoiseAudioDataset):
    """
    e.g:
    path: /datasets01/mls/mls_english/train/audio/9955/9413/9955_9413_000006.flac
    path_to_langid_fn: lang_dict[p.split("/")[3].split("_")[1]]
    """
    def __init__(
        self,
        langs_str,
        rand_factor,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            rand_factor=rand_factor,
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            num_buckets=num_buckets,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )
        # self.path_to_langid_fn = path_to_langid_fn
        self.langs = langs_str.split(",")
        self.lang_id_dict = {l: i for i, l in enumerate(self.langs)}
        self.lang_regex = r'|'.join(self.langs)

    def __getitem__(self, index):
        import soundfile as sf

        # path_or_fp = os.path.join(self.root_dir, str(self.fnames[index]))
        path_or_fp = str(self.fnames[index])
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        # NOTE: find lang in part
        langs = re.findall(self.lang_regex, _path)
        assert len(langs) == 1, f'{_path}'
        try:
            lang_id = self.lang_id_dict[langs[0]]
        except Exception as e:
            logger.warning(f'{langs}, {_path}')
        # lang_id = self.path_to_langid_fn(_path)
        return {
            "id": index, 
            "lang_id": [lang_id] * self.rand_factor,
            "source": [feats] * self.rand_factor,
        }

