import os
import tempfile
import re
import librosa
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
import math

from transformers import Wav2Vec2ForCTC, AutoProcessor

from lib import falign_ext

parser = argparse.ArgumentParser()
parser.add_argument("--uroman_txt", type=str)
parser.add_argument("--wav", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

ASR_SAMPLING_RATE = 16_000

MODEL_ID = "/upload/mms_zs"

processor = AutoProcessor.from_pretrained(args.model+MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(args.model+MODEL_ID)

token_file = args.model+"/upload/mms_zs/tokens.txt"

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    tokens = [x.strip() for x in open(token_file, "r").readlines()]

    txts = [x.strip() for x in open(args.uroman_txt, "r").readlines()]
    wavs = [x.strip() for x in open(args.wav, "r").readlines()]
    assert len(txts) == args.n * len(wavs)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    # clear it
    with open(args.dst + "/uasr_score", "w") as f1:
        pass

    for i, w in tqdm(enumerate(wavs)):
        assert isinstance(w, str)
        audio_samples = librosa.load(w, sr=ASR_SAMPLING_RATE, mono=True)[0]

        inputs = processor(
            audio_samples, sampling_rate=ASR_SAMPLING_RATE, return_tensors="pt"
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs).logits

        emissions = outputs.log_softmax(dim=-1).squeeze()
        
        for j in range(args.n):
            idx = (args.n * i) + j
            chars = txts[idx].split()
            token_sequence = [tokens.index(x) for x in chars]
        
            try:
                _, alphas, _ = falign_ext.falign(emissions, torch.tensor(token_sequence, device=device).int(), False)
                aligned_alpha = max(alphas[-1]).item()
            except:
                aligned_alpha = math.log(0.000000001)

            with open(args.dst + "/uasr_score", "a") as f1:
                f1.write(str(aligned_alpha) + "\n")
                f1.flush()