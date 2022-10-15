#!/usr/bin/env python

import argparse
import soundfile as sf
import torch

MODEL_ID_TO_ARGS = {
    "xm_transformer_unity_en-hk": {
        "beam": 10,
        "beam_mt": 10,
        "max_len_a": 0.2,
        "max_len_b": 200,
        "max_len_a_mt": 0,
        "max_len_b_mt": 200,
    },
    "xm_transformer_s2ut_en-hk": {
        "beam": 5,
        "max_len_a": 1,
        "max_len_b": 200,
    },
    "xm_transformer_unity_hk-en": {
        "beam": 10,
        "beam_mt": 10,
        "max_len_a": 0.3,
        "max_len_b": 200,
        "max_len_a_mt": 0,
        "max_len_b_mt": 200,
    },
    "xm_transformer_s2ut_hk-en": {
        "beam": 5,
        "max_len_a": 1,
        "max_len_b": 200,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, type=str, help="speech to unit model ids", choices=MODEL_ID_TO_ARGS.keys())
    parser.add_argument("--input-audio", required=True, type=str, help="input audio file")
    parser.add_argument("--output-audio", required=True, type=str, help="output audio file")
    args = parser.parse_args()

    tts_model = torch.hub.load(
        "pytorch/fairseq:ust",
        args.model_id,
        generation_args=MODEL_ID_TO_ARGS[args.model_id],
    )
    units, audio = tts_model.predict(args.input_audio, synthesize_speech=True)
    print(len(units.split(" ")))
    waveform, sample_rate = audio
    sf.write(args.output_audio, waveform.detach().cpu().numpy(), sample_rate)


if __name__ == '__main__':
    main()
