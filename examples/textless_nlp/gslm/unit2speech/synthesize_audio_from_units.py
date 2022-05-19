# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import soundfile as sf
from examples.textless_nlp.gslm.unit2speech.tts_data import (
    TacotronInputDataset,
)
from examples.textless_nlp.gslm.unit2speech.utils import (
    load_quantized_audio_from_file,
    load_tacotron,
    load_waveglow,
    synthesize_audio,
)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Wav2Vec 2.0 speech generator."
    )
    parser.add_argument(
        "--quantized_unit_path",
        type=str,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--tts_model_path",
        type=str,
        help="TTS model file path to use for inference",
    )
    parser.add_argument(
        "--waveglow_path",
        type=str,
        help="Path to the waveglow checkpoint (vocoder).",
    )
    parser.add_argument(
        "--code_dict_path",
        type=str,
        help="Code dict file path to use for inference",
    )
    parser.add_argument("--max_decoder_steps", type=int, default=2000)
    parser.add_argument("--denoiser_strength", type=float, default=0.1)
    parser.add_argument(
        "--out_audio_dir",
        type=str,
        help="Output directory to dump audio files",
    )

    return parser


def main(args, logger):
    # Load quantized audio
    logger.info(f"Loading quantized audio from {args.quantized_unit_path}...")
    names_batch, quantized_units_batch = load_quantized_audio_from_file(
        file_path=args.quantized_unit_path
    )

    logger.info(f"Loading TTS model from {args.tts_model_path}...")
    tacotron_model, sample_rate, hparams = load_tacotron(
        tacotron_model_path=args.tts_model_path,
        max_decoder_steps=args.max_decoder_steps,
    )

    logger.info(f"Loading Waveglow model from {args.waveglow_path}...")
    waveglow, denoiser = load_waveglow(waveglow_path=args.waveglow_path)

    if not os.path.exists(hparams.code_dict):
        hparams.code_dict = args.code_dict_path
    tts_dataset = TacotronInputDataset(hparams)

    for name, quantized_units in zip(names_batch, quantized_units_batch):
        quantized_units_str = " ".join(map(str, quantized_units))
        tts_input = tts_dataset.get_tensor(quantized_units_str)
        mel, aud, aud_dn, has_eos = synthesize_audio(
            tacotron_model,
            waveglow,
            denoiser,
            tts_input.unsqueeze(0),
            strength=args.denoiser_strength,
        )
        out_file_path = os.path.join(args.out_audio_dir, f"{name}.wav")
        sf.write(
            f"{out_file_path}", aud_dn[0].cpu().float().numpy(), sample_rate
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
