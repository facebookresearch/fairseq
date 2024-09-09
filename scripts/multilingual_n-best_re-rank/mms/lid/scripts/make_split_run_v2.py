import argparse
import json
from collections import defaultdict
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--dump', type=str)
    parser.add_argument('--splits', type=int)
    parser.add_argument('--set', type=str)
    args = parser.parse_args()

    for i in range(args.splits):
        tsv = args.dump + "/split_" + str(i) + "/split.tsv"
        if not os.path.exists(f"exp/mms1b_l4016/{args.set}/split_{i}"):
            os.makedirs(f"exp/mms1b_l4016/{args.set}/split_{i}")
        print(f"python infer.py /large_experiments/mms/data/bible/lid/manifest_mms_2/custom/vl+v3plus+grn_uvbg_a0.6_b0.7/ --path /large_experiments/mms/users/vineelkpratap/exps/wav2vec/lid/bible/vl+v3plus+grn_uvbg_a0.6_b0.7/bible_lang_f_n_b_dataset.max_tokens:1440000__model.mask_channel_prob:0.0__model.mask_prob:0__optimization.lr:[5e-06]__optimization.max_update:50000mms/checkpoints/checkpoint_best.pt --task audio_classification --infer-manifest {tsv} --output-path exp/mms1b_l4016/{args.set}/split_{i}")