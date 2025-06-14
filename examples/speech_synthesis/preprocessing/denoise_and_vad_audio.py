# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import torchaudio
try:
    import webrtcvad
except ImportError:
    raise ImportError("Please install py-webrtcvad: pip install webrtcvad")
import pandas as pd
from tqdm import tqdm

from examples.speech_synthesis.preprocessing.denoiser.pretrained import master64
import examples.speech_synthesis.preprocessing.denoiser.utils as utils
from examples.speech_synthesis.preprocessing.vad import (
    frame_generator, vad_collector, read_wave, write_wave, FS_MS, THRESHOLD,
    SCALE
)
from examples.speech_to_text.data_utils import save_df_to_tsv
from examples.speech_synthesis.preprocessing.tfgridnet.enh_inference import SeparateSpeech


log = logging.getLogger(__name__)

PATHS = ["after_denoise", "after_vad"]
MIN_T = 0.05


def generate_tmp_filename(extension="txt"):
    return tempfile._get_default_tempdir() + "/" + \
           next(tempfile._get_candidate_names()) + "." + extension


def convert_sr(inpath, sr, output_path=None):
    if not output_path:
        output_path = generate_tmp_filename("wav")
    cmd = f"sox {inpath} -r {sr} {output_path}"
    os.system(cmd)
    return output_path


def apply_vad(vad, inpath):
    audio, sample_rate = read_wave(inpath)
    frames = frame_generator(FS_MS, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, FS_MS, 300, vad, frames)
    merge_segments = list()
    timestamp_start = 0.0
    timestamp_end = 0.0
    # removing start, end, and long sequences of sils
    for i, segment in enumerate(segments):
        merge_segments.append(segment[0])
        if i and timestamp_start:
            sil_duration = segment[1] - timestamp_end
            if sil_duration > THRESHOLD:
                merge_segments.append(int(THRESHOLD / SCALE) * (b'\x00'))
            else:
                merge_segments.append(int((sil_duration / SCALE)) * (b'\x00'))
        timestamp_start = segment[1]
        timestamp_end = segment[2]
    segment = b''.join(merge_segments)
    return segment, sample_rate


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr, encoding="PCM_S",
                    bits_per_sample=16)


def process(args):
    # Making sure we are requested either denoise or vad
    if not args.denoise and not args.vad:
        log.error("No denoise or vad is requested.")
        return

    log.info("Creating out directories...")
    if args.denoise:
        out_denoise = Path(args.output_dir).absolute().joinpath(PATHS[0])
        out_denoise.mkdir(parents=True, exist_ok=True)
    if args.vad:
        out_vad = Path(args.output_dir).absolute().joinpath(PATHS[1])
        out_vad.mkdir(parents=True, exist_ok=True)

    # preparing the output dict
    output_dict = defaultdict(list)

    log.info(f"Parsing input manifest: {args.audio_manifest}")
    with open(args.audio_manifest, "r") as f:
        manifest_dict = csv.DictReader(f, delimiter="\t")
        for row in tqdm(manifest_dict):
           filename = str(row["audio"])

           final_output = filename
           keep_sample = True
           n_frames = row["n_frames"]
           snr = -1
           # Denoise
           if args.denoise:
              # Load pre-trained speech enhancement model and build VAD model
              log.info("Loading SeperateSpeech(TFGridnet) enhancement model...")
              if args.model == "SeparateSpeech":

                log.info(f"Training Configuration .yaml file: {args.config}")
                log.info(f"Pre-trained model .pth file: {args.pth_model}")
                model = SeparateSpeech(
                train_config = args.config, 
                model_file= args.pth_model,
                normalize_segment_scale=False,
                show_progressbar=True,
                ref_channel=4,
                normalize_output_wav=True)

                output_path_denoise = out_denoise.joinpath(Path(f"SeperateSpeech_{filename}").name)
                waveform, sr = torchaudio.load(filename)
                waveform = waveform.to("cpu")
                estimate = model(waveform)
                estimate = torch.tensor(estimate)
                torchaudio.save(output_path_denoise, estimate[0], 16_000, encoding="PCM_S", bits_per_sample=16)

              else:

                log.info("Loading pre-trained speech enhancement model...")
                model = master64().to(args.device)  
                # Set the output path for denoised audio
                output_path_denoise = out_denoise.joinpath(Path(f"master64_{filename}").name)

                # Convert to 16kHz if the sample rate is different
                tmp_path = convert_sr(final_output, 16000)
                # Load audio file and generate the enhanced version
                out, sr = torchaudio.load(tmp_path)
                out = out.to(args.device)
                estimate = model(out)
                estimate = (1 - args.dry_wet) * estimate + args.dry_wet * out
                write(estimate[0], str(output_path_denoise), sr)

                snr = utils.cal_snr(out, estimate)
                snr = snr.cpu().detach().numpy()[0][0]
                final_output = str(output_path_denoise)

           log.info("Building the VAD model...")
           vad = webrtcvad.Vad(int(args.vad_agg_level))

           if args.vad:
                output_path_vad = out_vad.joinpath(Path(filename).name)
                sr = torchaudio.info(final_output).sample_rate
                if sr in [16000, 32000, 48000]:
                    tmp_path = final_output
                elif sr < 16000:
                    tmp_path = convert_sr(final_output, 16000)
                elif sr < 32000:
                    tmp_path = convert_sr(final_output, 32000)
                else:
                    tmp_path = convert_sr(final_output, 48000)
                # apply VAD
                segment, sample_rate = apply_vad(vad, tmp_path)
                if len(segment) < sample_rate * MIN_T:
                     keep_sample = False
                     print((
                        f"WARNING: skip {filename} because it is too short "
                        f"after VAD ({len(segment) / sample_rate} < {MIN_T})"
                     ))
                else:
                    if sample_rate != sr:
                        tmp_path = generate_tmp_filename("wav")
                        write_wave(tmp_path, segment, sample_rate)
                        convert_sr(tmp_path, sr,
                               output_path=str(output_path_vad))
                    else:
                        write_wave(str(output_path_vad), segment, sample_rate)
                    final_output = str(output_path_vad)
                    segment, _ = torchaudio.load(final_output)
                    n_frames = segment.size(1)    

           if keep_sample:
                output_dict["id"].append(row["id"])
                output_dict["audio"].append(final_output)
                output_dict["n_frames"].append(n_frames)
                output_dict["tgt_text"].append(row["tgt_text"])
                output_dict["speaker"].append(row["speaker"])
                output_dict["src_text"].append(row["src_text"])
                output_dict["snr"].append(snr)     

        out_tsv_path = Path(args.output_dir) / Path(args.audio_manifest).name
        log.info(f"Saving manifest to {out_tsv_path.as_posix()}")
        save_df_to_tsv(pd.DataFrame.from_dict(output_dict), out_tsv_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest", "-i", required=True,
                        type=str, help="path to the input manifest.")
    parser.add_argument(
        "--output-dir", "-o", required=True, type=str,
        help="path to the output dir. it will contain files after denoising and"
             " vad"
    )
    parser.add_argument("--vad-agg-level", "-a", type=int, default=2,
                        help="the aggresive level of the vad [0-3].")
    parser.add_argument(
        "--dry-wet", "-dw", type=float, default=0.01,
        help="the level of linear interpolation between noisy and enhanced "
             "files."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="the device to be used for the speech enhancement model: "
             "cpu | cuda."
    )
    parser.add_argument("--denoise", action="store_true",
                        help="apply a denoising")
    parser.add_argument(
        "--model", "-m", type=str, default="master64",
        help="the speech enhancement model to be used: master64 | SeparateSpeech."
    )
    parser.add_argument("--config", type=str,
                        help="Training Configuration file for SeparateSpeech model.")
    parser.add_argument("--pth-model", type=str,
                        help="Path to the pre-trained model file for SeparateSpeech.")
    parser.add_argument("--vad", action="store_true", help="apply a VAD")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()