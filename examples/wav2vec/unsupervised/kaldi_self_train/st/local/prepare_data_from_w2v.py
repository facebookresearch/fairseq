import kaldi_io
import numpy as np
import os


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("w2v_dir", help="wav2vec feature and text directory")
    parser.add_argument("tar_root", help="output data directory in kaldi's format")
    parser.add_argument("split", help="name of the subset")
    parser.add_argument("--label", default="", help="if specified, copy labels too")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    tar_dir = os.path.join(args.tar_root, args.split)
    os.makedirs(tar_dir, exist_ok=True)

    lengths_path = os.path.join(args.w2v_dir, f"{args.split}.lengths")
    with open(lengths_path) as f:
        lengths = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengths[:-1]).tolist()
    feats = np.load(
        os.path.join(args.w2v_dir, f"{args.split}.npy"),
        mmap_mode="r"
    )
    assert feats.shape[0] == sum(lengths), \
        f"lengths mismatch {feats.shape[0]} != {sum(lengths)}"

    ark_path = os.path.join(tar_dir, "feats.ark")
    scp_path = os.path.join(tar_dir, "feats.scp")
    wspec = f"ark:| copy-feats --compress=true ark:- ark,scp:{ark_path},{scp_path}"
    with kaldi_io.open_or_fd(wspec, "wb") as f:
        for idx, (offset, length) in enumerate(zip(offsets, lengths)):
            feat = feats[offset:offset+length]
            kaldi_io.write_mat(f, feat, key=f"utt{idx:010d}")

    u2s_path = os.path.join(tar_dir, "utt2spk")
    s2u_path = os.path.join(tar_dir, "spk2utt")
    with open(u2s_path, "w") as f_u2s, open(s2u_path, "w") as f_s2u:
        for idx in range(len(lengths)):
            f_u2s.write(f"utt{idx:010d} utt{idx:010d}\n")
            f_s2u.write(f"utt{idx:010d} utt{idx:010d}\n")

    if bool(args.label):
        lab_path = os.path.join(args.w2v_dir, f"{args.split}.{args.label}")
        txt_path = os.path.join(tar_dir, "text")
        with open(lab_path) as f_lab, open(txt_path, "w") as f_txt:
            for idx, line in enumerate(f_lab):
                f_txt.write(f"utt{idx:010d} {line}")

if __name__ == "__main__":
    main()
