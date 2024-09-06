# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import ast
import argparse
import logging
import torch

from fairseq import utils
from fairseq.models.speech_dlm import SpeechDLM

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(in_file):
    with open(in_file) as f:
        data = [ast.literal_eval(line.strip()) for line in f]
    return data


def write_data(out_file, data):
    with open(out_file, 'w') as f:
        for d in data:
            f.write(str(d))
            f.write('\n')


def limit(codes, n):
    new_codes = {}
    for k, v in codes.items():
        new_codes[k] = ' '.join(v.split()[:n])
    return new_codes


def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available()

    # Load the data
    data = load_data(args.in_file)
    channels = args.channels.split(',')
    unit_sequences = [{
        channels[0]: d[channels[0]],
        channels[1]: d[channels[1]],
    } for d in data]
    fnames = [d['audio'] for d in data]
    print(f"Found {len(data)} sequences from {args.in_file}")

    # Limit the prefix size
    if args.prefix_size is not None:
        print(f"Limit the prefix size to {args.prefix_size}")
        unit_sequences = [limit(codes, args.prefix_size) for codes in unit_sequences]

    # Load model from ckpt
    print(f"Loading the SpeechDLM model from {args.ckpt}")
    model = SpeechDLM.from_pretrained(
                model_name_or_path=os.path.dirname(args.ckpt),
                checkpoint_file=os.path.basename(args.ckpt),
                data_name_or_path=args.data
            )
    model.eval()
    if use_cuda:
        model.cuda()

    # Set batch sizes
    model.cfg.dataset.max_tokens = args.batch_max_tokens
    model.max_positions = args.batch_max_positions
    if args.batch_max_sentences is not None:
        model.cfg.dataset.batch_size = args.batch_max_sentences

    # Set seed (if needed)
    if args.seed is not None:
        utils.set_torch_seed(args.seed)

    # Sample from the SpeechDLM model
    print(f"Generating {len(unit_sequences)} sequences with SpeechDLM model...\n"
          f"Generation args: sampling={(not args.beam_search)}, "
          f"sampling_topk={args.sampling_topk}, sampling_topp={args.sampling_topp}, "
          f"beam={args.beam_size}, min_len={args.min_len}, "
          f"max_len_a={args.max_len_a}, max_len_b={args.max_len_b}, "
          f"temperature={args.temperature}, dur_temperature={args.dur_temperature}, "
          f"seed={args.seed}")
    generated_units = model.sample(
            unit_sequences,
            sampling=(not args.beam_search),
            sampling_topk=args.sampling_topk,
            sampling_topp=args.sampling_topp,
            beam=args.beam_size,
            max_len_a=args.max_len_a,
            max_len_b=args.max_len_b,
            min_len=args.min_len,
            temperature=args.temperature,
            duration_temperature=args.dur_temperature,
            verbose=args.verbose,
            skip_invalid_size_inputs=args.skip_invalid_size_batch,
        )

    # Create the generated sequences
    generated_data = []
    for fname, gen_units in zip(fnames, generated_units):
        d = {
            "audio" : fname+'-generated',
            **gen_units
        }
        generated_data.append(d)

    # Write the generated sequences
    print(f"Write the generated units to {args.out_file}")
    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    write_data(args.out_file, generated_data)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file",
        type=str,
        required=True,
        help="Input file following the same format of the output from create_input.py",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to the model data dir (containing dict files)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        required=True,
        help="Path of the output file.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default='unitA,unitB',
        help="Comma-separated list of the channel names"
             "(Default: 'unitA,unitB').",
    )
    parser.add_argument("--prefix-size", type=int, default=None,
                        help='Limit the prefix size')

    # Batch sizes
    parser.add_argument("--batch-max-tokens", type=int, default=9216,
                        help='maximum number of tokens considered in a batch')
    parser.add_argument("--batch-max-positions", type=int, default=6144,
                        help='maximum number of tokens allowed for a sentence in a batch')
    parser.add_argument("--batch-max-sentences", type=int, default=None,
                        help='maximum number of sentences considered in a batch')
    parser.add_argument("--skip-invalid-size-batch", action='store_true',
                        help='skip sentences with more tokens than --batch-max-positions')

    # Generation args
    parser.add_argument("--beam-search", action='store_true',
                        help='perform beam search instead of sampling')
    parser.add_argument("--beam-size", type=int, default=5,
                        help="beam width (used in both sampling and beam search mode) "
                        "(default: 5)")
    parser.add_argument("--sampling-topk", type=int, default=-1,
                        help="only sample from top-k candidates (default: -1, non applied)")
    parser.add_argument("--sampling-topp", type=float, default=-1.0,
                        help="only sample among the smallest set of elements whose cumulative "
                        "probability mass exceeds p (default: -1.0, non applied)")
    parser.add_argument("--max-len-a", type=int, default=0,
                        help="generate sequences of maximum length ax + b, "
                        "where x is the source length (default: 0)")
    parser.add_argument("--max-len-b", type=int, default=500,
                        help="generate sequences of maximum length ax + b, "
                        "where x is the source length (default: 500 ~ 10s)")
    parser.add_argument("--min-len", type=int, default=1,
                        help="generate sequences of maximum length ax + b, "
                        "where x is the source length (default: 1)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature when generating unit tokens (default: 1.0)")
    parser.add_argument("--dur-temperature", type=float, default=1.0,
                        help="temperature when generating duration tokens (default: 1.0)")
    parser.add_argument("--verbose", action='store_true',
                        help="print the scores given by the model to generated sequences")
    parser.add_argument("--seed", type=int, default=123,
                        help="seed of the generation model")

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli_main()
