import argparse
import json
from collections import defaultdict
import os
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()

    # grab asr refs from raw babel manifests for min 10 sec version (from wavs file)

    manifests = [args.src + "/" + d + "/eval_manifest.json" for d in os.listdir(args.src)]
    wav_to_ref = {}
    for m in manifests:
        lines = open(m, "r").readlines()
        for l in lines:
            data = json.loads(l)
            wav_to_ref[data['audio_filepath']] = data['text']

    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]
    refs = []
    for wav in wavs:
        wav = wav.replace("//", "/")
        if wav not in wav_to_ref:
            print("Missing ref?")
            import pdb;pdb.set_trace()
        refs.append(wav_to_ref[wav] + "\n")

    with open(args.dst + "/test.wrd", "w") as f:
        f.writelines(refs)
    
    with open(args.dst + "/test.ltr", "w") as f:
        f.writelines([" ".join(x.replace(" ", "|")) for x in refs])
