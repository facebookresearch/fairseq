# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Optional

from examples.speech_to_text.data_utils import S2TDataConfigWriter


def gen_config_yaml(
    manifest_root: Path,
    yaml_filename: str = "config.yaml",
    specaugment_policy: Optional[str] = "lb",
    feature_transform: Optional[List[str]] = None,
    input_channels: Optional[int] = 1,
    input_feat_per_channel: Optional[int] = 80,
    audio_root: str = "",
    vocoder_type: Optional[str] = None,
    vocoder_checkpoint: Optional[str] = None,
    vocoder_cfg: Optional[str] = None,
    extra=None,
):
    manifest_root = manifest_root.absolute()
    writer = S2TDataConfigWriter(manifest_root / yaml_filename)

    if input_channels is not None:
        writer.set_input_channels(input_channels)
    if input_feat_per_channel is not None:
        writer.set_input_feat_per_channel(input_feat_per_channel)
    specaugment_setters = {
        "lb": writer.set_specaugment_lb_policy,
        "ld": writer.set_specaugment_ld_policy,
        "sm": writer.set_specaugment_sm_policy,
        "ss": writer.set_specaugment_ss_policy,
    }
    specaugment_setter = specaugment_setters.get(specaugment_policy, None)
    if specaugment_setter is not None:
        specaugment_setter()

    if feature_transform is None:
        feature_transform = []
    else:
        writer.set_feature_transforms("*", feature_transform)

    if specaugment_policy is not None:
        writer.set_feature_transforms("_train", feature_transform + ["specaugment"])

    if len(audio_root) > 0:
        writer.set_audio_root(audio_root)

    if (
        vocoder_type is not None
        and vocoder_checkpoint is not None
        and vocoder_cfg is not None
    ):
        writer.set_extra(
            {
                "vocoder": {
                    "type": vocoder_type,
                    "config": vocoder_cfg,
                    "checkpoint": vocoder_checkpoint,
                }
            }
        )

    if extra is not None:
        writer.set_extra(extra)
    writer.flush()


def load_units(in_file):
    out = {}
    with open(in_file) as f:
        for line in f:
            sample_id, units = line.strip().split("|", 1)
            out[sample_id] = units.split()

    return out


def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out
