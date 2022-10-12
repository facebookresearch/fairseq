import os
import argparse
import torch
import torchaudio
import torchaudio.functional as F
from datasets import load_dataset
from examples.speech_matrix.data_helper.data_cfg import FLEURS_LANGS


def get_lang_data(lang, out_audio_dir, out_manifest_dir, split, out_sr=16000):
    lang_code = lang[:2]
    data = load_dataset("fleurs", lang)
    data_size = len(data[split])
    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(out_manifest_dir, exist_ok=True)

    # audio, nframes
    save_split = split
    if split == "validation":
        save_split = "valid"
    aud_manifest = os.path.join(out_manifest_dir, f"{save_split}_{lang_code}.tsv")
    out_aud_manifest = open(aud_manifest, "w")
    out_aud_manifest.write(out_audio_dir + "\n")

    trans_fn = os.path.join(out_manifest_dir, f"{save_split}_{lang_code}.trans")
    out_trans = open(trans_fn, "w")

    raw_trans_fn = os.path.join(out_manifest_dir, f"{save_split}_{lang_code}.raw.trans")
    out_raw_trans = open(raw_trans_fn, "w")

    for idx in range(data_size):
        # aud: {'path', 'array', 'sampling_rate'}
        aud = data[split][idx]["audio"]
        waveform = torch.Tensor(aud["array"].reshape(1, len(aud["array"])))
        in_sr = aud["sampling_rate"]
        # resample audio
        if in_sr != out_sr:
            F.resample(waveform, in_sr, out_sr)
        nf = waveform.size(-1)
        fn = ".".join(os.path.basename(aud["path"]).split(".")[:-1])
        save_fn = os.path.join(out_audio_dir, fn + ".flac")
        # save audio
        if not os.path.exists(save_fn):
            torchaudio.save(save_fn, waveform, out_sr)

        out_aud_manifest.write(f"{fn}.flac\t{nf}\n")
        text = data[split][idx]["transcription"]
        raw_text = data[split][idx]["raw_transcription"]
        out_trans.write(text + "\n")
        out_raw_trans.write(raw_text + "\n")
    out_aud_manifest.close()
    out_trans.close()
    out_raw_trans.close()
    print(f"save to {aud_manifest}")
    print(f"save to {trans_fn}")
    print(f"save to {raw_trans_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FLEURS testset preparation")
    parser.add_argument("--proc-fleurs-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.proc_fleurs_dir, exist_ok=True)

    for lang in FLEURS_LANGS:
        lang_code = lang[:2]
        for split in ["validation", "test"]:
            get_lang_data(
                lang,
                out_audio_dir=os.path.join(args.proc_fleurs_dir, "audios", lang_code),
                out_manifest_dir=os.path.join(args.proc_fleurs_dir, "aud_manifests"),
                split=split,
                out_sr=16000,
            )
