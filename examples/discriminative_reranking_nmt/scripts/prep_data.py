#!/usr/bin/env python

import argparse
from multiprocessing import Pool
from pathlib import Path

import sacrebleu
import sentencepiece as spm


def read_text_file(filename):
    with open(filename, "r") as f:
        output = [line.strip() for line in f]

    return output


def get_bleu(in_sent, target_sent):
    bleu = sacrebleu.corpus_bleu([in_sent], [[target_sent]])
    out = " ".join(
        map(str, [bleu.score, bleu.sys_len, bleu.ref_len] + bleu.counts + bleu.totals)
    )
    return out


def get_ter(in_sent, target_sent):
    ter = sacrebleu.corpus_ter([in_sent], [[target_sent]])
    out = " ".join(map(str, [ter.score, ter.num_edits, ter.ref_length]))
    return out


def init(sp_model):
    global sp
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)


def process(source_sent, target_sent, hypo_sent, metric):
    source_bpe = " ".join(sp.EncodeAsPieces(source_sent))
    hypo_bpe = [" ".join(sp.EncodeAsPieces(h)) for h in hypo_sent]

    if metric == "bleu":
        score_str = [get_bleu(h, target_sent) for h in hypo_sent]
    else:  # ter
        score_str = [get_ter(h, target_sent) for h in hypo_sent]

    return source_bpe, hypo_bpe, score_str


def main(args):
    assert (
        args.split.startswith("train") or args.num_shards == 1
    ), "--num-shards should be set to 1 for valid and test sets"
    assert (
        args.split.startswith("train")
        or args.split.startswith("valid")
        or args.split.startswith("test")
    ), "--split should be set to train[n]/valid[n]/test[n]"

    source_sents = read_text_file(args.input_source)
    target_sents = read_text_file(args.input_target)

    num_sents = len(source_sents)
    assert num_sents == len(
        target_sents
    ), f"{args.input_source} and {args.input_target} should have the same number of sentences."

    hypo_sents = read_text_file(args.input_hypo)
    assert (
        len(hypo_sents) % args.beam == 0
    ), f"Number of hypotheses ({len(hypo_sents)}) cannot be divided by beam size ({args.beam})."

    hypo_sents = [
        hypo_sents[i : i + args.beam] for i in range(0, len(hypo_sents), args.beam)
    ]
    assert num_sents == len(
        hypo_sents
    ), f"{args.input_hypo} should contain {num_sents * args.beam} hypotheses but only has {len(hypo_sents) * args.beam}. (--beam={args.beam})"

    output_dir = args.output_dir / args.metric
    for ns in range(args.num_shards):
        print(f"processing shard {ns+1}/{args.num_shards}")
        shard_output_dir = output_dir / f"split{ns+1}"
        source_output_dir = shard_output_dir / "input_src"
        hypo_output_dir = shard_output_dir / "input_tgt"
        metric_output_dir = shard_output_dir / args.metric

        source_output_dir.mkdir(parents=True, exist_ok=True)
        hypo_output_dir.mkdir(parents=True, exist_ok=True)
        metric_output_dir.mkdir(parents=True, exist_ok=True)

        if args.n_proc > 1:
            with Pool(
                args.n_proc, initializer=init, initargs=(args.sentencepiece_model,)
            ) as p:
                output = p.starmap(
                    process,
                    [
                        (source_sents[i], target_sents[i], hypo_sents[i], args.metric)
                        for i in range(ns, num_sents, args.num_shards)
                    ],
                )
        else:
            init(args.sentencepiece_model)
            output = [
                process(source_sents[i], target_sents[i], hypo_sents[i], args.metric)
                for i in range(ns, num_sents, args.num_shards)
            ]

        with open(source_output_dir / f"{args.split}.bpe", "w") as s_o, open(
            hypo_output_dir / f"{args.split}.bpe", "w"
        ) as h_o, open(metric_output_dir / f"{args.split}.{args.metric}", "w") as m_o:
            for source_bpe, hypo_bpe, score_str in output:
                assert len(hypo_bpe) == len(score_str)
                for h, m in zip(hypo_bpe, score_str):
                    s_o.write(f"{source_bpe}\n")
                    h_o.write(f"{h}\n")
                    m_o.write(f"{m}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-source", type=Path, required=True)
    parser.add_argument("--input-target", type=Path, required=True)
    parser.add_argument("--input-hypo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--beam", type=int, required=True)
    parser.add_argument("--sentencepiece-model", type=str, required=True)
    parser.add_argument("--metric", type=str, choices=["bleu", "ter"], default="bleu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n-proc", type=int, default=8)

    args = parser.parse_args()

    main(args)
