"""
Generic dataset that loads data from LMDB database.
"""
import warnings
from pathlib import Path
from typing import List, Optional, Union

import lmdb
import torch
from einops import rearrange
from torchvision.io import decode_image, write_video


class LMDBLoader(object):
    """
    Generic dataset that loads data from LMDB database.
    """
    def __init__(
        self,
        lmdb_path: str,
        load_stride: int = 1,
        load_float16: bool = False,
        load_type: str = "feats",
        verbose: bool = False,
        lmdb_window_size: int = 16,
        lmdb_stride: int = 2,
        feat_dim: Optional[int] = None,
    ):
        """
        Args:
            lmdb_path (Union[str, Path]): Path to LMDB database.
            load_stride (int): Stride for loading frames from LMDB database.
            load_float16 (bool): Whether to load frames as float16.
            load_type (str): Type of data to load from LMDB database.
                Either "feats" or "frames" or "pseudo-labels".
            verbose (bool): Whether to print verbose messages.
            lmdb_window_size (int): Window size for sliding window approach.
                Only required if load_type == "feats" or "pseudo-labels".
            lmdb_stride (int): Stride for sliding window approach.
                Only required if load_type == "feats" or "pseudo-labels".
            feat_dim (Optional[int]): Feature dimensionality.
                Only required if load_type == "feats".
        """
        assert load_type in ["feats", "frames", "pseudo-labels"], \
            f"load_type must be either 'feats' or 'frames' or 'pseudo-labels', but got {load_type}."
        self.lmdb_path = lmdb_path
        self.load_stride = load_stride
        self.load_float16 = load_float16
        self.lmdb = self._init_lmdb()
        self.load_type = load_type
        self.verbose = verbose
        if self.load_type == "feats":
            assert feat_dim is not None, "feat_dim must be provided if load_type == 'feats'."
            self.vid_feat_dim = feat_dim
        if self.load_type in ["feats", "pseudo-labels"]:
            self.lmdb_window_size = lmdb_window_size
            self.lmdb_stride = lmdb_stride

    def _init_lmdb(self) -> lmdb.Environment:
        """Initialise LMDB database."""
        return lmdb.open(self.lmdb_path, readonly=True, lock=False, max_readers=10000)

    @staticmethod
    def _get_feat_key(episode_name: str, frame_index: int, suffix: str = ".np") -> bytes:
        """Returns key for features in LMDB database."""
        key_end = f"{frame_index + 1:07d}{suffix}"
        return f"{Path(episode_name.split('.')[0]).stem}/{key_end}".encode('ascii')

    @staticmethod
    def _get_pseudo_label_key(
        episode_name: str, frame_index: int, suffix: str = ".np"
    ) -> List[bytes]:
        """Returns key for pseudo-labels and corresponding probabilities in LMDB database."""
        key_end = f"{frame_index + 1:07d}{suffix}"
        return f"{Path(episode_name.split('.')[0] + '_label').stem}/{key_end}".encode('ascii'), \
            f"{Path(episode_name.split('.')[0] + '_prob').stem}/{key_end}".encode('ascii')

    @staticmethod
    def _get_rbg_key(
        episode_name: str, frame_index: int, suffix: str = ".jpg") -> bytes:
        """Returns key for RGB frames in LMDB database."""
        key_end = f"{frame_index + 1:07d}{suffix}"
        return f"{Path(episode_name.split('.')[0]).stem}/{key_end}".encode('ascii')

    def feature_idx_to_frame_idx(
        self,
        feature_idx: int,
    ) -> int:
        """Convert feature index to frame index."""
        begin_idx = self.lmdb_window_size // 2 - 1
        return begin_idx + feature_idx * self.lmdb_stride

    def frame_idx_to_feature_idx(
        self,
        frame_idx: int,
    ) -> int:
        """
        Convert frame index to feature index.
        Formula: frame_idx = begin_idx + feature_idx * stride
            with begin_idx = self.lmdb_window_size // 2 - 1
        """
        begin_idx = self.lmdb_window_size // 2 - 1
        return max(0, (frame_idx - begin_idx) // self.lmdb_stride)

    def load_sequence(
        self, episode_name: str, begin_frame: int, end_frame: int,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Loads a sequence of frames/features/pseudo-labels from LMDB database."""
        if self.load_type == "feats" or self.load_type == "pseudo-labels":
            begin_frame = self.frame_idx_to_feature_idx(frame_idx=begin_frame)
            end_frame = self.frame_idx_to_feature_idx(frame_idx=end_frame)
            if self.load_type == "feats":
                all_feats = []
            else:
                all_labels, all_probs = [], []
        else:
            # load RGB frames
            frames = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for frame_idx in range(begin_frame, end_frame, self.load_stride):
                # get key for LMDB database
                if self.load_type == "feats":
                    feats_key = self._get_feat_key(episode_name, frame_idx)
                    with self.lmdb.begin() as txn:
                        features = torch.zeros((self.vid_feat_dim,), dtype=torch.float16)
                        try:
                            # print(feats_key)
                            features = torch.frombuffer(txn.get(feats_key), dtype=torch.float16)
                        except (KeyError, TypeError, ValueError):
                            if self.verbose:
                                print(f"Key {feats_key} not found in LMDB database.")
                        all_feats.append(features)
                elif self.load_type == "pseudo-labels":
                    label_key, prob_key = self._get_pseudo_label_key(episode_name, frame_idx)
                    with self.lmdb.begin() as txn:
                        labels = torch.zeros((5,), dtype=torch.long)
                        probs = torch.zeros((5,), dtype=torch.float16)
                        try:
                            labels = torch.frombuffer(txn.get(label_key), dtype=torch.long)
                            probs = torch.frombuffer(txn.get(prob_key), dtype=torch.float16)
                        except (KeyError, TypeError, ValueError):
                            if self.verbose:
                                print(f"Key {label_key} or {prob_key} not found in LMDB database.")
                        all_labels.append(labels)
                        all_probs.append(probs)
                else:
                    # load rgb frames
                    rgb_key = self._get_rbg_key(episode_name, frame_idx)
                    with self.lmdb.begin() as txn:
                        frame = torch.zeros((3, 256, 256), dtype=torch.uint8)
                        try:
                            frame = decode_image(
                                torch.frombuffer(txn.get(rgb_key), dtype=torch.uint8)
                            )
                        except (KeyError, TypeError, ValueError):
                            if self.verbose:
                                print(f"Key {rgb_key} not found in LMDB database.")
                        frames.append(frame)

        if self.load_type == "feats":
            if self.load_float16:
                return torch.stack(all_feats).half()
            try:
                all_feats = torch.stack(all_feats).float()
            except:
                print(len(all_feats), episode_name, begin_frame, end_frame)
            return all_feats
        elif self.load_type == "pseudo-labels":
            if self.load_float16:
                return torch.stack(all_labels), torch.stack(all_probs).half()
            return torch.stack(all_labels), torch.stack(all_probs).float()
        else:
            # load rgb frames + rearrange from (T, C, H, W) to (T, H, W, C)
            return rearrange(torch.stack(frames), "t c h w -> t h w c")

    def save_rgb_video(
        self,
        episode_name: str,
        begin_frame: int, end_frame: int,
        save_dir: str,
    ) -> None:
        """Function to save RGB frames as video."""
        # load frames
        frames = self.load_sequence(episode_name, begin_frame, end_frame)
        # save video
        write_video(
            filename=f"{save_dir}{episode_name}_{begin_frame}-{end_frame}.mp4",
            video_array=frames,
            fps=25,
        )
