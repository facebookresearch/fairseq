import os
import argparse
import torchaudio
import torchaudio.functional as F
from examples.speech_matrix.data_helper.data_cfg import EP_LANGS, manifest_key
from examples.speech_matrix.data_helper.cleaners import text_cleaners
from examples.speech_matrix.data_helper.test_data_helper import gen_manifest


domain = "epst"


def extract_and_downsample_segment(
    aud_path, start_time, end_time, seg_path, out_sr=16000
):
    if not os.path.exists(seg_path):
        in_sr = torchaudio.info(aud_path).sample_rate
        start_frame = int(in_sr * start_time)
        end_frame = int(in_sr * end_time)
        num_frames = end_frame - start_frame
        wav, _ = torchaudio.load(
            aud_path, frame_offset=start_frame, num_frames=num_frames
        )
        # downsampled to 16kHz
        ds_wav = F.resample(wav, in_sr, out_sr)
        # save segments
        torchaudio.save(seg_path, ds_wav, sample_rate=out_sr)
    metadata = torchaudio.info(seg_path)
    assert metadata.sample_rate == out_sr
    num_channels = metadata.num_channels
    if num_channels > 1:
        tmp_path = ".".join(seg_path.split(".")[:-1] + ["tmp", "wav"])
        os.system(f"mv {seg_path} {tmp_path}")
        os.system(f"sox {tmp_path} -c 1 -r {out_sr} {seg_path}")
        os.system(f"rm {tmp_path}")
    return torchaudio.info(seg_path).num_frames


def segment_epst_aud(
    epst_dir, src_lang, tgt_lang, proc_epst_dir, out_sr=16000, domain="epst"
):
    src_aud_dir = os.path.join(epst_dir, src_lang, "audios")
    out_aud_dir = os.path.join(proc_epst_dir, "audios", src_lang)
    # tmp dir to convert .m4a to .wav
    tmp_out_aud_dir = os.path.join(out_aud_dir, "tmp")
    out_tsv_dir = os.path.join(
        proc_epst_dir, "aud_manifests", src_lang + "-" + tgt_lang
    )
    os.makedirs(out_aud_dir, exist_ok=True)
    os.makedirs(tmp_out_aud_dir, exist_ok=True)
    os.makedirs(out_tsv_dir, exist_ok=True)

    for split in ["test"]:
        tsv_fn = os.path.join(
            out_tsv_dir, "_".join([split, domain, src_lang, tgt_lang]) + ".tsv"
        )
        tsv_out = open(tsv_fn, "w")
        tsv_out.write(out_aud_dir + "\n")

        # segment.lst: <audio_file> <start> <end>
        # seg_id: <audio_file>_<start>_<end>
        seg_fn = os.path.join(epst_dir, src_lang, tgt_lang, split, "segments.lst")
        with open(seg_fn, "r") as fin:
            for line in fin:
                aud, st, et = line.strip().split()
                aud_path = os.path.join(src_aud_dir, aud + ".m4a")
                tmp_aud_path = os.path.join(tmp_out_aud_dir, aud + ".wav")
                if not os.path.exists(tmp_aud_path):
                    os.system("ffmpeg -i {} {}".format(aud_path, tmp_aud_path))

                seg_fn = "_".join([aud, st, et]) + ".wav"
                seg_path = os.path.join(out_aud_dir, seg_fn)
                nframes = extract_and_downsample_segment(
                    tmp_aud_path, float(st), float(et), seg_path, out_sr=16000
                )
                tsv_out.write(seg_fn + "\t" + str(nframes) + "\n")
        tsv_out.close()


def normalize_translations(epst_dir, src_lang, tgt_lang, manifest_dir, domain="epst"):
    in_tsv_dir = os.path.join(epst_dir, src_lang, tgt_lang)
    # for split in ["train", "dev", "test"]:
    for split in ["test"]:
        split_dir = os.path.join(in_tsv_dir, split)
        in_trans_fn = os.path.join(split_dir, "segments." + tgt_lang)
        out_trans_fn = os.path.join(manifest_dir, f"{split}_{domain}.{tgt_lang}")
        fin = open(in_trans_fn, "r")
        fout = open(out_trans_fn, "w")
        for line in fin:
            proc_line = text_cleaners(line.strip(), tgt_lang)
            fout.write(proc_line + "\n")
        fin.close()
        fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("EuroParl-ST testset preparation")
    parser.add_argument("--epst-dir", type=str, required=True)
    parser.add_argument("--proc-epst-dir", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.proc_epst_dir, exist_ok=True)

    for src_lang in EP_LANGS:
        for tgt_lang in EP_LANGS:
            if src_lang == tgt_lang:
                continue
            print(f"processing EPST audios: {src_lang}-{tgt_lang}...")
            segment_epst_aud(
                args.epst_dir,
                src_lang,
                tgt_lang,
                args.proc_epst_dir,
                out_sr=16000,
                domain=domain,
            )

            print(f"Generating manifests: {src_lang}-{tgt_lang}...")
            aud_manifest_fn = os.path.join(
                args.proc_epst_dir,
                "aud_manifests",
                f"{src_lang}-{tgt_lang}",
                f"test_{domain}_{src_lang}_{tgt_lang}.tsv",
            )
            manifest_dir = os.path.join(
                args.save_root, manifest_key, f"{src_lang}-{tgt_lang}"
            )
            s2u_manifest_fn = os.path.join(manifest_dir, "test_epst.tsv")
            asr_manifest_fn = os.path.join(manifest_dir, "source_unit", "test_epst.tsv")
            os.makedirs(os.path.join(manifest_dir, "source_unit"), exist_ok=True)

            gen_manifest(
                aud_manifest_fn,
                s2u_manifest_fn,
                asr_manifest_fn,
            )
            normalize_translations(
                args.epst_dir, src_lang, tgt_lang, manifest_dir, domain=domain
            )
