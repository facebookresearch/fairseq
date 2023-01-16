import json
import os
import tqdm
from fairseq.data import Dictionary, data_utils


def load_dictionary(dict_path):
    return Dictionary.load(dict_path)

def load_dataset(split_path, src_dict):
    dataset = data_utils.load_indexed_dataset(
        split_path,
        src_dict,
        combine=False,  # set to true for loading `train*`
    )
    if dataset is None:
        raise FileNotFoundError(f"Dataset not found: {split_path}")
    return dataset

def load_bpe(enc_path):
    with open(enc_path) as f:
        bpe2idx = json.load(f)
        idx2bpe = {v: k for k, v in bpe2idx.items()}
    return bpe2idx, idx2bpe

def detokenize(tokens, src_dict, idx2bpe):
    raw_inds = map(int, src_dict.string(tokens).split())
    raw_chrs = "".join([idx2bpe[raw_ind] for raw_ind in raw_inds])
    raw_chrs = raw_chrs.replace("\u0120", " ")
    return raw_chrs

def _main(src_root, src_dict_path, src_bpe_path, src_splits, tgt_root, tgt_splits):
    src_dict = load_dictionary(src_dict_path)
    bpe2idx, idx2bpe = load_bpe(src_bpe_path)

    assert len(src_splits) == len(tgt_splits)
    for src_split, tgt_split in zip(src_splits, tgt_splits):
        src_dataset = load_dataset(f"{src_root}/{src_split}", src_dict)
        tgt_path = f"{tgt_root}/{tgt_split}.txt"
        print(f"processing {src_split} (dump to {tgt_path})...")
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        with open(tgt_path, "w") as f:
            for tokens in tqdm.tqdm(src_dataset):
                raw_str = detokenize(tokens, src_dict, idx2bpe)
                f.write(raw_str + "\n")

def main_pt():
    src_root = "/datasets01/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin/121219/bookwiki_CC-NEWS_openwebtext_stories-mmap2-bin"
    src_dict_path = f"{src_root}/dict.txt"
    src_bpe_path = f"{src_root}/encoder.json"
    src_splits = [
        "bookwiki_aml-mmap2-bin/shard0/train",
        "bookwiki_aml-mmap2-bin/shard1/train",
        "bookwiki_aml-mmap2-bin/shard2/train",
        "bookwiki_aml-mmap2-bin/shard3/train",
        "bookwiki_aml-mmap2-bin/shard4/train",
        "bookwiki_aml-mmap2-bin/valid/valid",
    ]

    tgt_root = "/checkpoint/wnhsu/data/data2vec2/data/text/bookwiki_aml-full-mmap2-txt"
    tgt_splits = [
        "train0",
        "train1",
        "train2",
        "train3",
        "train4",
        "valid",
    ]
    _main(src_root, src_dict_path, src_bpe_path, src_splits, tgt_root, tgt_splits)

def main_ft():
    src_root = "/fsx-wav2vec/wnhsu/data/data2vec2/data/text/GLUE"
    src_dict_path = f"{src_root}/dict.txt"
    src_bpe_path = f"{src_root}/encoder.json"
    src_splits = [
        "CoLA-bin/input0/train",
        "CoLA-bin/input0/valid",
        "CoLA-bin/input0/test",

        "MNLI-bin/input0/train",
        "MNLI-bin/input0/valid",
        "MNLI-bin/input0/test",
        "MNLI-bin/input0/test1",
        "MNLI-bin/input1/train",
        "MNLI-bin/input1/valid",
        "MNLI-bin/input1/test",
        "MNLI-bin/input1/test1",

        "MRPC-bin/input0/train",
        "MRPC-bin/input0/valid",
        "MRPC-bin/input0/test",
        "MRPC-bin/input1/train",
        "MRPC-bin/input1/valid",
        "MRPC-bin/input1/test",

        "QNLI-bin/input0/train",
        "QNLI-bin/input0/valid",
        "QNLI-bin/input0/test",
        "QNLI-bin/input1/train",
        "QNLI-bin/input1/valid",
        "QNLI-bin/input1/test",

        "QQP-bin/input0/train",
        "QQP-bin/input0/valid",
        "QQP-bin/input0/test",
        "QQP-bin/input1/train",
        "QQP-bin/input1/valid",
        "QQP-bin/input1/test",

        "RTE-bin/input0/train",
        "RTE-bin/input0/valid",
        "RTE-bin/input0/test",
        "RTE-bin/input1/train",
        "RTE-bin/input1/valid",
        "RTE-bin/input1/test",

        "SST-2-bin/input0/train",
        "SST-2-bin/input0/valid",
        "SST-2-bin/input0/test",

        "STS-B-bin/input0/train",
        "STS-B-bin/input0/valid",
        "STS-B-bin/input0/test",
        "STS-B-bin/input1/train",
        "STS-B-bin/input1/valid",
        "STS-B-bin/input1/test",
    ]

    tgt_root = "/fsx-wav2vec/wnhsu/data/data2vec2/data/text/GLUE_chr"
    tgt_splits = [
        "CoLA-bin/input0/train",
        "CoLA-bin/input0/valid",
        "CoLA-bin/input0/test",

        "MNLI-bin/input0/train",
        "MNLI-bin/input0/valid",
        "MNLI-bin/input0/test",
        "MNLI-bin/input0/test1",
        "MNLI-bin/input1/train",
        "MNLI-bin/input1/valid",
        "MNLI-bin/input1/test",
        "MNLI-bin/input1/test1",

        "MRPC-bin/input0/train",
        "MRPC-bin/input0/valid",
        "MRPC-bin/input0/test",
        "MRPC-bin/input1/train",
        "MRPC-bin/input1/valid",
        "MRPC-bin/input1/test",

        "QNLI-bin/input0/train",
        "QNLI-bin/input0/valid",
        "QNLI-bin/input0/test",
        "QNLI-bin/input1/train",
        "QNLI-bin/input1/valid",
        "QNLI-bin/input1/test",

        "QQP-bin/input0/train",
        "QQP-bin/input0/valid",
        "QQP-bin/input0/test",
        "QQP-bin/input1/train",
        "QQP-bin/input1/valid",
        "QQP-bin/input1/test",

        "RTE-bin/input0/train",
        "RTE-bin/input0/valid",
        "RTE-bin/input0/test",
        "RTE-bin/input1/train",
        "RTE-bin/input1/valid",
        "RTE-bin/input1/test",

        "SST-2-bin/input0/train",
        "SST-2-bin/input0/valid",
        "SST-2-bin/input0/test",

        "STS-B-bin/input0/train",
        "STS-B-bin/input0/valid",
        "STS-B-bin/input0/test",
        "STS-B-bin/input1/train",
        "STS-B-bin/input1/valid",
        "STS-B-bin/input1/test",
    ]
    _main(src_root, src_dict_path, src_bpe_path, src_splits, tgt_root, tgt_splits)


if __name__ == "__main__":
    main_pt()
    main_ft()
