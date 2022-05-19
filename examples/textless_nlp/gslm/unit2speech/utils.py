# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from examples.textless_nlp.gslm.unit2speech.tacotron2.model import Tacotron2
from examples.textless_nlp.gslm.unit2speech.tacotron2.waveglow_denoiser import (
    Denoiser,
)


def load_quantized_audio_from_file(file_path):
    base_fname_batch, quantized_units_batch = [], []
    with open(file_path) as f:
        for line in f:
            base_fname, quantized_units_str = line.rstrip().split("|")
            quantized_units = [int(q) for q in quantized_units_str.split(" ")]
            base_fname_batch.append(base_fname)
            quantized_units_batch.append(quantized_units)
    return base_fname_batch, quantized_units_batch


def synthesize_audio(model, waveglow, denoiser, inp, lab=None, strength=0.0):
    assert inp.size(0) == 1
    inp = inp.cuda()
    if lab is not None:
        lab = torch.LongTensor(1).cuda().fill_(lab)

    with torch.no_grad():
        _, mel, _, ali, has_eos = model.inference(inp, lab, ret_has_eos=True)
        aud = waveglow.infer(mel, sigma=0.666)
        aud_dn = denoiser(aud, strength=strength).squeeze(1)
    return mel, aud, aud_dn, has_eos


def load_tacotron(tacotron_model_path, max_decoder_steps):
    ckpt_dict = torch.load(tacotron_model_path)
    hparams = ckpt_dict["hparams"]
    hparams.max_decoder_steps = max_decoder_steps
    sr = hparams.sampling_rate
    model = Tacotron2(hparams)
    model.load_state_dict(ckpt_dict["model_dict"])
    model = model.cuda().eval().half()
    return model, sr, hparams


def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)["model"]
    waveglow = waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser
