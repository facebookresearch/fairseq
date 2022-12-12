import logging
import argparse
import random
import sys
import os
import numpy as np
import torch
import soundfile as sf
import shutil
import librosa
import json
from pathlib import Path
from tqdm import tqdm
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

dir_path = os.path.dirname(__file__)
resynth_path = os.path.dirname(os.path.abspath(__file__)) + "/speech-resynthesis"
sys.path.append(resynth_path)

from models import CodeGenerator
from inference import scan_checkpoint, load_checkpoint, generate
from emotion_models.pitch_predictor import load_ckpt as load_pitch_predictor
from emotion_models.duration_predictor import load_ckpt as load_duration_predictor
from dataset import load_audio, MAX_WAV_VALUE, parse_style, parse_speaker, EMOV_SPK2ID, EMOV_STYLE2ID


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('debug.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_generation_file(fname):
    lines = open(fname).read()
    lines = lines.split('\n')

    results = {}
    for l in lines:
        if len(l) == 0:
            continue

        if l[0] == 'H':
            parts = l[2:].split('\t')
            if len(parts) == 2:
                sid, utt = parts
            else:
                sid, _, utt = parts
            sid = int(sid)
            utt = [int(x) for x in utt.split()]
            if sid in results:
                results[sid]['H'] = utt
            else:
                results[sid] = {'H': utt}
        elif l[0] == 'S':
            sid, utt = l[2:].split('\t')
            sid = int(sid)
            utt = [x for x in utt.split()]
            if sid in results:
                results[sid]['S'] = utt
            else:
                results[sid] = {'S': utt}
        elif l[0] == 'T':
            sid, utt = l[2:].split('\t')
            sid = int(sid)
            utt = [int(x) for x in utt.split()]
            if sid in results:
                results[sid]['T'] = utt
            else:
                results[sid] = {'T': utt}

    for d, result in results.items():
        if 'H' not in result:
            result['H'] = result['S']

    return results


def get_code_to_fname(manifest, tokens):
    if tokens is None:
        code_to_fname = {}
        with open(manifest) as f:
            for line in f:
                line = line.strip()
                fname, code = line.split()
                code = code.replace(',', ' ')
                code_to_fname[code] = fname

        return code_to_fname

    with open(manifest) as f:
        fnames = [l.strip() for l in f.readlines()]
        root = Path(fnames[0])
        fnames = fnames[1:]
        if '\t' in fnames[0]:
            fnames = [x.split()[0] for x in fnames]

    with open(tokens) as f:
        codes = [l.strip() for l in f.readlines()]

    code_to_fname = {}
    for fname, code in zip(fnames, codes):
        code = code.replace(',', ' ')
        code_to_fname[code] = str(root / fname)

    return root, code_to_fname


def code_to_str(s):
    k = ' '.join([str(x) for x in s])
    return k


def get_praat_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0


def generate_from_code(generator, h, code, spkr=None, f0=None, gst=None, device="cpu"):
    batch = {
        'code': torch.LongTensor(code).to(device).view(1, -1),
    }
    if spkr is not None:
        batch['spkr'] = spkr.to(device).unsqueeze(0)
    if f0 is not None:
        batch['f0'] = f0.to(device)
    if gst is not None:
        batch['style'] = gst.to(device)

    with torch.no_grad():
        audio, rtf = generate(h, generator, batch)
        audio = librosa.util.normalize(audio / 2 ** 15)

    return audio


@torch.no_grad()
def synth(argv, interactive=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-path', type=Path, help='Translation Model Output', required=True)
    parser.add_argument('--data', type=Path, help='a directory with the files: src.tsv, src.km, trg.tsv, trg.km, orig.tsv, orig.km')
    parser.add_argument("--orig-tsv", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/data.tsv")
    parser.add_argument("--orig-km", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/core_manifests/emov_16khz_km_100/data.km")

    parser.add_argument('--checkpoint-file', type=Path, help='Generator Checkpoint', required=True)
    parser.add_argument('--dur-model', type=Path, help='a token duration prediction model (if tokens were deduped)')
    parser.add_argument('--f0-model', type=Path, help='a f0 prediction model')

    parser.add_argument('-s', '--src-emotion', default=None)
    parser.add_argument('-t', '--trg-emotion', default=None)
    parser.add_argument('-N', type=int, default=10)
    parser.add_argument('--split', default="test")

    parser.add_argument('--outdir', type=Path, default=Path('results'))
    parser.add_argument('--orig-filename', action='store_true')

    parser.add_argument('--device', type=int, default=0)
    a = parser.parse_args(argv)

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(a.device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    dur_models = {
        "neutral":   load_duration_predictor(f"{a.dur_model}/neutral.ckpt"),
        "amused":    load_duration_predictor(f"{a.dur_model}/amused.ckpt"),
        "disgusted": load_duration_predictor(f"{a.dur_model}/disgusted.ckpt"),
        "angry":     load_duration_predictor(f"{a.dur_model}/angry.ckpt"),
        "sleepy":    load_duration_predictor(f"{a.dur_model}/sleepy.ckpt"),
    }
    logger.info(f"loaded duration prediction model from {a.dur_model}")

    f0_model = load_pitch_predictor(a.f0_model).to(a.device)
    logger.info(f"loaded f0 prediction model from {a.f0_model}")

    # we need to know how to map code back to the filename
    # (if we want the original files names as output)
    results = parse_generation_file(a.result_path)
    _, src_code_to_fname = get_code_to_fname(f'{a.data}/files.{a.split}.{a.src_emotion}', f'{a.data}/{a.split}.{a.src_emotion}')
    _, tgt_code_to_fname = get_code_to_fname(f'{a.data}/files.{a.split}.{a.trg_emotion}', f'{a.data}/{a.split}.{a.trg_emotion}')

    # we need the originals (before dedup) to get the ground-truth durations
    orig_tsv = open(a.orig_tsv, 'r').readlines()
    orig_tsv_root, orig_tsv = orig_tsv[0].strip(), orig_tsv[1:]
    orig_km = open(a.orig_km, 'r').readlines()
    fname_to_idx = {orig_tsv_root + "/" + line.split("\t")[0]: i for i, line in enumerate(orig_tsv)}

    outdir = a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / '0-source').mkdir(exist_ok=True)
    (outdir / '1-src-tokens-src-style-src-f0').mkdir(exist_ok=True)
    (outdir / '2-src-tokens-trg-style-src-f0').mkdir(exist_ok=True)
    (outdir / '2.5-src-tokens-trg-style-src-f0').mkdir(exist_ok=True)
    (outdir / '3-src-tokens-trg-style-pred-f0').mkdir(exist_ok=True)
    (outdir / '4-gen-tokens-trg-style-pred-f0').mkdir(exist_ok=True)
    (outdir / '5-target').mkdir(exist_ok=True)

    N = 0
    results = list(results.items())
    random.shuffle(results)
    for i, (sid, result) in tqdm(enumerate(results)):
        N += 1
        if N > a.N and a.N != -1:
            break

        if '[' in result['S'][0]:
            result['S'] = result['S'][1:]
        if '_' in result['S'][-1]:
            result['S'] = result['S'][:-1]
        src_ref = src_code_to_fname[code_to_str(result['S'])]
        trg_ref = tgt_code_to_fname[code_to_str(result['T'])]

        src_style, trg_style = None, None
        src_spkr, trg_spkr = None, None
        src_f0 = None
        src_audio = (load_audio(src_ref)[0] / MAX_WAV_VALUE) * 0.95
        trg_audio = (load_audio(trg_ref)[0] / MAX_WAV_VALUE) * 0.95
        src_audio = torch.FloatTensor(src_audio).unsqueeze(0).cuda()
        trg_audio = torch.FloatTensor(trg_audio).unsqueeze(0).cuda()

        src_spkr = parse_speaker(src_ref, h.multispkr)
        src_spkr = src_spkr if src_spkr in EMOV_SPK2ID else random.choice(list(EMOV_SPK2ID.keys()))
        src_spkr = EMOV_SPK2ID[src_spkr]
        src_spkr = torch.LongTensor([src_spkr])
        trg_spkr = parse_speaker(trg_ref, h.multispkr)
        trg_spkr = trg_spkr if trg_spkr in EMOV_SPK2ID else random.choice(list(EMOV_SPK2ID.keys()))
        trg_spkr = EMOV_SPK2ID[trg_spkr]
        trg_spkr = torch.LongTensor([trg_spkr])

        src_style = EMOV_STYLE2ID[a.src_emotion]
        src_style = torch.LongTensor([src_style]).cuda()
        trg_style_str = a.trg_emotion
        trg_style = EMOV_STYLE2ID[a.trg_emotion]
        trg_style = torch.LongTensor([trg_style]).cuda()

        src_tokens = list(map(int, orig_km[fname_to_idx[src_ref]].strip().split(" ")))
        src_tokens = torch.LongTensor(src_tokens).unsqueeze(0)
        src_tokens_dur_pred = torch.LongTensor(list(map(int, result['S']))).unsqueeze(0)
        src_tokens_dur_pred = dur_models[trg_style_str].inflate_input(src_tokens_dur_pred)
        gen_tokens = torch.LongTensor(result['H']).unsqueeze(0)
        gen_tokens = dur_models[trg_style_str].inflate_input(gen_tokens)
        trg_tokens = torch.LongTensor(result['T']).unsqueeze(0)
        trg_tokens = dur_models[trg_style_str].inflate_input(trg_tokens)

        src_f0 = get_praat_f0(src_audio.unsqueeze(0).cpu().numpy())
        src_f0 = torch.FloatTensor(src_f0).cuda()

        pred_src_f0 = f0_model.inference(torch.LongTensor(src_tokens).to(a.device), src_spkr, trg_style).unsqueeze(0)
        pred_src_dur_pred_f0 = f0_model.inference(torch.LongTensor(src_tokens_dur_pred).to(a.device), src_spkr, trg_style).unsqueeze(0)
        pred_gen_f0 = f0_model.inference(torch.LongTensor(gen_tokens).to(a.device), src_spkr, trg_style).unsqueeze(0)
        pred_trg_f0 = f0_model.inference(torch.LongTensor(trg_tokens).to(a.device), src_spkr, trg_style).unsqueeze(0)

        if a.orig_filename:
            path = src_code_to_fname[code_to_str(result['S'])]
            sid = str(sid) + "__" + Path(path).stem
        shutil.copy(src_code_to_fname[code_to_str(result['S'])], outdir / '0-source' / f'{sid}.wav')

        audio = generate_from_code(generator, h, src_tokens, spkr=src_spkr, f0=src_f0, gst=src_style, device=a.device)
        sf.write(outdir / '1-src-tokens-src-style-src-f0' / f'{sid}.wav', audio, samplerate=h.sampling_rate)

        audio = generate_from_code(generator, h, src_tokens, spkr=src_spkr, f0=src_f0, gst=trg_style, device=a.device)
        sf.write(outdir / '2-src-tokens-trg-style-src-f0' / f'{sid}.wav', audio, samplerate=h.sampling_rate)

        audio = generate_from_code(generator, h, src_tokens_dur_pred, spkr=src_spkr, f0=src_f0, gst=trg_style, device=a.device)
        sf.write(outdir / '2.5-src-tokens-trg-style-src-f0' / f'{sid}.wav', audio, samplerate=h.sampling_rate)

        audio = generate_from_code(generator, h, src_tokens_dur_pred, spkr=src_spkr, f0=pred_src_dur_pred_f0, gst=trg_style, device=a.device)
        sf.write(outdir / '3-src-tokens-trg-style-pred-f0' / f'{sid}.wav', audio, samplerate=h.sampling_rate)

        audio = generate_from_code(generator, h, gen_tokens, spkr=src_spkr, f0=pred_gen_f0, gst=trg_style, device=a.device)
        sf.write(outdir / '4-gen-tokens-trg-style-pred-f0' / f'{sid}.wav', audio, samplerate=h.sampling_rate)

        shutil.copy(tgt_code_to_fname[code_to_str(result['T'])], outdir / '5-target' / f'{sid}.wav')

    logger.info("Done.")


if __name__ == '__main__':
    synth(sys.argv[1:])
