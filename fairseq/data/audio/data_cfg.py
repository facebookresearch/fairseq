# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Dict, Optional


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML: pip install PyYAML")
        self.config = {}
        if yaml_path.is_file():
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                raise Exception(
                    f"Failed to load config from {yaml_path.as_posix()}: {e}"
                )
        else:
            raise FileNotFoundError(f"{yaml_path.as_posix()} not found")
        self.root = yaml_path.parent

    def _auto_convert_to_abs_path(self, x):
        if isinstance(x, str):
            if not Path(x).exists() and (self.root / x).exists():
                return (self.root / x).as_posix()
        elif isinstance(x, dict):
            return {k: self._auto_convert_to_abs_path(v) for k, v in x.items()}
        return x

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def speaker_set_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("speaker_set_filename", None)

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("pre_tokenizer", {"tokenizer": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        tokenizer = self.config.get("bpe_tokenizer", {"bpe": None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sample_rate(self):
        return self.config.get("sample_rate", 16_000)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)

    @property
    def use_sample_rate(self):
        """Needed by the dataset loader to see if the model requires
        raw audio with specific sample rate as inputs."""
        return self.config.get("use_sample_rate", 16000)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    def get_feature_transforms(self, split, is_train):
        """Split-specific feature transforms. Allowing train set
        wildcard `_train`, evaluation set wildcard `_eval` and general
        wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get("transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        cfg["transforms"] = cur
        return cfg

    @property
    def global_cmvn_stats_npz(self) -> Optional[str]:
        path = self.config.get("global_cmvn", {}).get("stats_npz_path", None)
        return self._auto_convert_to_abs_path(path)

    @property
    def vocoder(self) -> Optional[Dict[str, str]]:
        return self.config.get("vocoder", None)
