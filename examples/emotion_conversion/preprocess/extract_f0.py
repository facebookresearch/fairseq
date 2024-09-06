import argparse
from tqdm import tqdm
from multiprocessing import Manager, Pool

from scipy.io.wavfile import read
from librosa.util import normalize
import numpy as np
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

MAX_WAV_VALUE = 32768.0

parser = argparse.ArgumentParser(description="")
parser.add_argument("tsv", help="")
parser.add_argument("--extractor", choices=["crepe", "pyaapt"], default="pyaapt", help="")
parser.add_argument("--interp", action="store_true", help="")
parser.add_argument("--n_workers", type=int, default=40, help="")
args = parser.parse_args()

tsv_lines = open(args.tsv, "r").readlines()
root, tsv_lines = tsv_lines[0].strip(), tsv_lines[1:]


def extract_f0(tsv_line):
    wav_path, _ = tsv_line.split("\t")
    wav_path = root.strip() + "/" + wav_path
    sr, wav = read(wav_path)
    wav = wav / MAX_WAV_VALUE
    wav = normalize(wav) * 0.95

    if args.extractor == "pyaapt":
        frame_length = 20.0
        pad = int(frame_length / 1000 * sr) // 2
        wav = np.pad(wav.squeeze(), (pad, pad), "constant", constant_values=0)
        signal = basic.SignalObj(wav, sr)
        pitch = pYAAPT.yaapt(
                signal,
                **{
                    'frame_length': frame_length,
                    'frame_space': 5.0,
                    'nccf_thresh1': 0.25,
                    'tda_frame_length': 25.0
                })
        pitch = pitch.samp_interp[None, None, :] if args.interp else pitch.samp_values[None, None, :]
        pitch = pitch[0, 0]
        f0_path = wav_path.replace(".wav", ".yaapt")
        f0_path += ".interp.f0" if args.interp else ".f0"
        np.save(f0_path, pitch)


def main():
    with Pool(args.n_workers) as p:
        r = list(tqdm(p.imap(extract_f0, tsv_lines), total=len(tsv_lines)))


if __name__ == "__main__":
    main()
