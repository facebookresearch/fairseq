# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

from fairseq.data import Dictionary


def get_config_from_yaml(yaml_path: Path):
    try:
        import yaml
    except ImportError:
        print("Please install PyYAML: pip install PyYAML")
    config = {}
    if yaml_path.is_file():
        try:
            with open(yaml_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            raise Exception(f"Failed to load config from {yaml_path.as_posix()}: {e}")
    else:
        raise FileNotFoundError(f"{yaml_path.as_posix()} not found")

    return config


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        self.config = get_config_from_yaml(yaml_path)
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
        """speaker set file under data root"""
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

    def standardize_audio(self) -> bool:
        return self.use_audio_input and self.config.get("standardize_audio", False)

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
    def vocoder(self) -> Dict[str, str]:
        vocoder = self.config.get("vocoder", {"type": "griffin_lim"})
        return self._auto_convert_to_abs_path(vocoder)

    @property
    def hub(self) -> Dict[str, str]:
        return self.config.get("hub", {})


class S2SDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", None)

    @property
    def pre_tokenizer(self) -> Dict:
        return None

    @property
    def bpe_tokenizer(self) -> Dict:
        return None

    @property
    def input_transformed_channels(self):
        """The number of channels in the audio after feature transforms"""
        # TODO: move this into individual transforms
        _cur = self.config.get("transforms", {})
        cur = _cur.get("_train", [])

        _channels = self.input_channels
        if "delta_deltas" in cur:
            _channels *= 3

        return _channels

    @property
    def output_sample_rate(self):
        """The audio sample rate of output target speech"""
        return self.config.get("output_sample_rate", 22050)

    @property
    def target_speaker_embed(self):
        """Target speaker embedding file (one line per target audio sample)"""
        return self.config.get("target_speaker_embed", None)


class MultitaskConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        config = get_config_from_yaml(yaml_path)
        self.config = {}
        for k, v in config.items():
            self.config[k] = SingleTaskConfig(k, v)

    def get_all_tasks(self):
        return self.config

    def get_single_task(self, name):
        assert name in self.config, f"multitask '{name}' does not exist!"
        return self.config[name]


class SingleTaskConfig(object):
    def __init__(self, name, config):
        self.task_name = name
        self.config = config
        dict_path = config.get("dict", "")
        self.tgt_dict = Dictionary.load(dict_path) if Path(dict_path).exists() else None

    @property
    def data(self):
        return self.config.get("data", "")

    @property
    def decoder_type(self):
        return self.config.get("decoder_type", "transformer")

    @property
    def decoder_args(self):
        """Decoder arch related args"""
        args = self.config.get("decoder_args", {})
        return Namespace(**args)

    @property
    def criterion_cfg(self):
        """cfg for the multitask criterion"""
        if self.decoder_type == "ctc":
            from fairseq.criterions.ctc import CtcCriterionConfig

            cfg = CtcCriterionConfig
            cfg.zero_infinity = self.config.get("zero_infinity", True)
        else:
            from fairseq.criterions.label_smoothed_cross_entropy import (
                LabelSmoothedCrossEntropyCriterionConfig,
            )

            cfg = LabelSmoothedCrossEntropyCriterionConfig
            cfg.label_smoothing = self.config.get("label_smoothing", 0.2)
        return cfg

    @property
    def input_from(self):
        """Condition on encoder/decoder of the main model"""
        return "decoder" if "decoder_layer" in self.config else "encoder"

    @property
    def input_layer(self):
        if self.input_from == "decoder":
            return self.config["decoder_layer"] - 1
        else:
            # default using the output from the last encoder layer (-1)
            return self.config.get("encoder_layer", 0) - 1

    @property
    def loss_weight_schedule(self):
        return (
            "decay"
            if "loss_weight_max" in self.config
            and "loss_weight_decay_steps" in self.config
            else "fixed"
        )

    def get_loss_weight(self, num_updates):
        if self.loss_weight_schedule == "fixed":
            weight = self.config.get("loss_weight", 1.0)
        else:  # "decay"
            assert (
                self.config.get("loss_weight_decay_steps", 0) > 0
            ), "loss_weight_decay_steps must be greater than 0 for a decay schedule"
            loss_weight_min = self.config.get("loss_weight_min", 0.0001)
            loss_weight_decay_stepsize = (
                self.config["loss_weight_max"] - loss_weight_min
            ) / self.config["loss_weight_decay_steps"]
            weight = max(
                self.config["loss_weight_max"]
                - loss_weight_decay_stepsize * num_updates,
                loss_weight_min,
            )
        return weight
