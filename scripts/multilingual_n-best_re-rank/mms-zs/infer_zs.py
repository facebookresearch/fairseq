import os
import tempfile
import re
import librosa
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse

from transformers import Wav2Vec2ForCTC, AutoProcessor
from torchaudio.models.decoder import ctc_decoder

uroman_dir = "uroman"
assert os.path.exists(uroman_dir)
UROMAN_PL = os.path.join(uroman_dir, "bin", "uroman.pl")

ASR_SAMPLING_RATE = 16_000

MODEL_ID = "upload/mms_zs"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

token_file = "upload/mms_zs/tokens.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()
    
    tokens = [x.strip() for x in open(token_file, "r").readlines()]

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

    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=tokens,
        beam_size=1,
        sil_score=0,
        blank_token="<s>",
    )

    wavs = [x.strip() for x in open(args.wavs, "r").readlines()]

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    # clear it
    with open(args.dst + "/pron", "w") as f1, open(args.dst + "/pron_no_sp", "w") as f2:
        pass

    for w in tqdm(wavs):
        assert isinstance(w, str)
        audio_samples = librosa.load(w, sr=ASR_SAMPLING_RATE, mono=True)[0]

        inputs = processor(
            audio_samples, sampling_rate=ASR_SAMPLING_RATE, return_tensors="pt"
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs).logits

        beam_search_result = beam_search_decoder(outputs.to("cpu"))
        
        transcription = [tokens[x] for x in beam_search_result[0][0].tokens]
        pron = " ".join(transcription)
        pron_no_sp = " ".join([x for x in transcription if x != "|"])
        
        with open(args.dst + "/pron", "a") as f1, open(args.dst + "/pron_no_sp", "a") as f2:
            f1.write(pron + "\n")
            f2.write(pron_no_sp + "\n")
            f1.flush()
            f2.flush()