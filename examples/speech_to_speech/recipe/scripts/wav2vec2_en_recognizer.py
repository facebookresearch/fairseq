# Copyright (c) Carnegie Mellon University (Jiatong Shi)
#
# # This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import argparse
import librosa
import os
import soundfile as sf
import torch

NUMBER_SIZE = 6

def pad_zero(number):
    return "0" * (NUMBER_SIZE - len(number)) + number

def recognizer(utt_fn, processor, model):
    
    audio_input, sample_rate = sf.read(utt_fn)
    if sample_rate != 16000:
        audio_input = librosa.resample(audio_input, sample_rate, 16000)
    input_values = processor([audio_input], return_tensors="pt", padding="longest").input_values

    # retrieve logits
    logits = model(input_values).logits
    
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


def main(args):
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    output = open(args.recognized_output, "w", encoding="utf-8")
    hyp_list = os.listdir(args.result_path)
    for hyp in hyp_list:
        if not hyp.endswith(args.extension):
            continue
        hyp_id = hyp[:-len(args.extension)]
        if hyp_id.isnumeric():
            hyp_id = pad_zero(hyp_id)
        recog_hyp = recognizer(os.path.join(args.result_path, hyp), processor, model)
        output.write("{} {}\n".format(hyp_id, recog_hyp))
    
    output.close()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audios", type=str, required=True, help="test audios for inference"
    )
    parser.add_argument(
        "--extension", type=str, required=True, help="extension of file for filter"
    )
    parser.add_argument(
        "--recognized_output", tyep=str, default="recognized.txt", help="output for recognition results"
    )
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()