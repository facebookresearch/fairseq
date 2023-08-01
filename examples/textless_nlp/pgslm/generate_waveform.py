# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import argparse
import json
import logging
from pathlib import Path
import soundfile as sf
import torch

from tqdm import tqdm

from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_result(args, data, sample_id, pred_wav):
    assert "audio" in data or args.results_path is not None
    if args.results_path:
        fname = Path(data["audio"]).name if "audio" in data else f"{sample_id}_pred.wav"
        out_file = Path(args.results_path) / fname

    sf.write(
        out_file.as_posix(),
        pred_wav.detach().cpu().numpy(),
        args.sample_rate,
    )


def load_data(in_file):
    with open(in_file) as f:
        data = [ast.literal_eval(line.strip()) for line in f]

    return data


def get_f0_upsample_ratio(code_hop_size, f_hop_size):
    ratio = (code_hop_size // 160) // (f_hop_size // 256) * 2
    return ratio


def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
    if use_cuda:
        vocoder = vocoder.cuda()

    data = load_data(args.in_file)

    if args.results_path:
        Path(args.results_path).mkdir(exist_ok=True, parents=True)

    for i, d in tqdm(enumerate(data), total=len(data)):
        code_key = "cpc_km100" if "cpc_km100" in d else "hubert"
        code = list(map(int, d[code_key].split()))

        x = {
            "code": torch.LongTensor(code).view(1, -1),
            "f0": torch.Tensor(d["f0"]).view(1, -1),
        }

        f0_up_ratio = get_f0_upsample_ratio(
            vocoder_cfg["code_hop_size"], vocoder_cfg["hop_size"]
        )
        if f0_up_ratio > 1:
            bsz, cond_length = x["f0"].size()
            x["f0"] = x["f0"].unsqueeze(2).repeat(1, 1, f0_up_ratio).view(bsz, -1)

        x = utils.move_to_cuda(x) if use_cuda else x
        wav = vocoder(x)
        dump_result(args, d, i, wav)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file",
        type=str,
        required=True,
        help="Input file following the same format of the output from sample.py ('f0' and 'cpc_km100/hubert' are required fields)",
    )
    parser.add_argument(
        "--vocoder", type=str, required=True, help="path to the vocoder"
    )
    parser.add_argument(
        "--vocoder-cfg",
        type=str,
        required=True,
        help="path to the vocoder config",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Output directory. If not set, the audios will be stored following the 'audio' field specified in the input file.",
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
