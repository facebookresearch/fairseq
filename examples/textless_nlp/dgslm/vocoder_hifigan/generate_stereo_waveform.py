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
        fname = Path(data["audio"]).stem + ".wav" if "audio" in data else f"{sample_id}_pred.wav"
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


def load_vocoder(vocoder_path, vocoder_cfg_path, use_cuda=True):
    with open(vocoder_cfg_path) as f:
        cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(vocoder_path, cfg).eval()
    if use_cuda:
        vocoder = vocoder.cuda()
    return vocoder


def code2wav(vocoder, code, speaker_id, use_cuda=True):
    if isinstance(code, str):
        code = list(map(int, code.split()))
    inp = dict()
    inp["code"] = torch.LongTensor(code).view(1, -1)
    if vocoder.model.multispkr:
        inp["spkr"] = torch.LongTensor([speaker_id]).view(1, 1)
    if use_cuda:
        inp = utils.move_to_cuda(inp)
    return vocoder(inp)


def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    vocoder = load_vocoder(args.vocoder, args.vocoder_cfg, use_cuda)

    data = load_data(args.in_file)

    if args.results_path:
        Path(args.results_path).mkdir(exist_ok=True, parents=True)

    channels = args.channels.split(',')
    speakers = [args.channel1_spk, args.channel2_spk]

    for i, d in tqdm(enumerate(data), total=len(data)):
        wavs = []
        for key, speaker_id in zip(channels, speakers):
            wav = code2wav(vocoder, d[key], speaker_id, use_cuda=use_cuda)
            wavs.append(wav)

        wav = torch.stack(wavs, dim=-1)
        if args.mix:
            wav = torch.mean(wav, dim=-1)

        dump_result(args, d, i, wav)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file",
        type=str,
        required=True,
        help="Input file following the same format of the output from create_input.py",
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
    parser.add_argument(
        "--channels",
        type=str,
        default='unitA,unitB',
        help="Comma-separated list of the channel names"
             "(Default: 'unitA,unitB').",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Output directory. If not set, the audios will be stored following the 'audio' field specified in the input file",
    )
    parser.add_argument("--channel1-spk", type=int, default=0, help="Speaker of the first channel",)
    parser.add_argument("--channel2-spk", type=int, default=4, help="Speaker of the second channel",)
    parser.add_argument("--mix", action="store_true", help="Mix the two channels to create output mono files")
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
