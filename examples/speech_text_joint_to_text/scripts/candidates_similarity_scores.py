# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import pickle
import sys

import torch
from torch.nn import functional as F

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.logging.meters import StopwatchMeter

DEBUG = False


def similarity_matrix(teacher_states, student_states, teacher_masking, student_masking, eps=1e-6):
    x = teacher_states.transpose(0, 1)  # from T X B X D to B X T X D
    y = student_states.transpose(0, 1)
    x = x / (x.norm(dim=2, keepdim=True) + eps)
    y = y / (y.norm(dim=2, keepdim=True) + eps)
    # lengths: batch X seqLen
    sim_scores_xy = torch.bmm(x, y.transpose(1, 2))  # batch X lenx X leny ]
    if y.dtype == torch.float16:
        sim_scores_xy = sim_scores_xy.float()
        y = y.float()
        x = x.float()
    if teacher_masking != []:
        assert len(teacher_masking) == 1
        sim_scores_xy = sim_scores_xy.masked_fill(
            teacher_masking[0].unsqueeze(-1), float("-inf")
        )
    if student_masking != []:
        sim_scores_xy = sim_scores_xy.masked_fill(
            student_masking[0].unsqueeze(1), float("-inf")
        )
    return sim_scores_xy


# To avoid recomputing textual encoder outputs
# for each speech sample
TEXT_CACHE = {}


def score(simmatrix, wstart, wend, audio_start, audio_len):
    area = simmatrix[audio_start:audio_start+audio_len, wstart:wend]
    return area.mean().item() * math.log(audio_len)


def scores_for_words(simmatrix, remaining_word_boundatries, starting_idxs, score_by_start_idx):
    if len(remaining_word_boundatries) <= 1:
        return score_by_start_idx
    wstart, wend = remaining_word_boundatries[0], remaining_word_boundatries[1]
    curr_w_len = remaining_word_boundatries[1] - remaining_word_boundatries[0]
    possible_lens = range(curr_w_len, curr_w_len * 3)
    possible_scores = [
        (score(simmatrix, wstart, wend, si, pl) + score_by_start_idx[si][0], (si, si + pl))
        for si in starting_idxs for pl in possible_lens if si + pl < simmatrix.shape[0]]
    if len(possible_scores) == 0:
        return {}
    next_scores_by_start_idx = {}
    for ps in possible_scores:
        next_start_idx = ps[1][1]
        if next_start_idx in next_scores_by_start_idx and next_scores_by_start_idx[next_start_idx][0] > ps[0]:
            continue
        new_list = score_by_start_idx[ps[1][0]][1].copy()
        new_list.append(ps[1])
        next_scores_by_start_idx[next_start_idx] = (ps[0], new_list)
    return scores_for_words(simmatrix, remaining_word_boundatries[1:], list(next_scores_by_start_idx.keys()), next_scores_by_start_idx)


def score_from_simmatrix(sim_matrix, candidate_txt):
    words_boundaries = []
    phonemes = candidate_txt.split(" ")
    for ph_idx in range(len(phonemes)):
        if phonemes[ph_idx].startswith("▁"):
            words_boundaries.append(ph_idx)

    words_boundaries.append(len(phonemes))
    start_idxs = list(range(sim_matrix.shape[0] - len(phonemes) - 1))
    score_by_start_idx = {si: (0, []) for si in start_idxs}
    scores = list(scores_for_words(sim_matrix, words_boundaries, start_idxs, score_by_start_idx).values())
    max_score = 0
    candidate = None
    for s in scores:
        if s[0] > max_score:
            max_score = s[0]
            candidate = s
    if DEBUG:
        print(candidate)
    return max_score


def generate_scores(model, sample, candidate_list, src_dict, use_cuda):
    encoder = model.encoder
    with torch.no_grad():
        speech_encoder_outs = encoder.spch_encoder(
            sample['src_tokens'], sample['src_lengths'], return_all_hiddens=True)
        speech_encoder_outs = encoder.process_attentive_loss_states(
            speech_encoder_outs,
            speech_encoder_outs["encoder_states"][
                -encoder.cross_attentive_loss_before_last_layer - 1
            ],
        )
        scores = []
        for candidate in candidate_list:
            if candidate in TEXT_CACHE:
                text_encoder_outs = TEXT_CACHE[candidate]
            else:
                src_txt_tokens = src_dict.encode_line(
                    candidate, add_if_not_exist=False, append_eos=True).long()
                src_txt_tokens = fairseq_data_utils.collate_tokens(
                    [src_txt_tokens],
                    src_dict.pad(),
                    src_dict.eos(),
                    left_pad=False,
                    move_eos_to_beginning=False,
                )
                src_txt_lengths = torch.tensor([src_txt_tokens.size()[0]], dtype=torch.long)
                if use_cuda:
                    src_txt_lengths = utils.move_to_cuda(src_txt_lengths)
                    src_txt_tokens = utils.move_to_cuda(src_txt_tokens)
                text_encoder_outs = encoder.text_encoder(
                    src_txt_tokens, src_txt_lengths, return_all_hiddens=True)
                text_encoder_outs = encoder.process_attentive_loss_states(
                    text_encoder_outs,
                    text_encoder_outs["encoder_states"][
                        -encoder.cross_attentive_loss_before_last_layer - 1
                    ],
                )
                TEXT_CACHE[candidate] = text_encoder_outs
            sim_matrix = similarity_matrix(
                teacher_states=speech_encoder_outs["encoder_states"],
                student_states=text_encoder_outs["encoder_states"],
                teacher_masking=speech_encoder_outs["encoder_padding_mask"],
                student_masking=text_encoder_outs["encoder_padding_mask"],
            ).squeeze(0)
            sim_matrix = sim_matrix[:, :-1]  # remove eos
            # score = sim_matrix.mean(dim=1).max().item()
            score = score_from_simmatrix(sim_matrix, candidate)
            scores.append(score)
    return scores


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert args.results_path is not None
    _main(args)


def _main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairseq_cli.candidates_similarity_score')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Load candidates
    logger.info('loading candidate list from {}'.format(args.list_candidates))
    with open(args.list_candidates) as f:
        candidate_list = [l.strip() for l in f]

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    dataset = task.dataset(args.gen_subset)
    progress = progress_bar.progress_bar(
        range(len(dataset)),
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    num_sentences = 0
    scores = {}
    for sample_id in progress:
        gen_timer.start()
        sample = dataset.collater([dataset[sample_id]])
        sample = utils.move_to_cuda(sample['net_input']) if use_cuda else sample['net_input']
        scores[sample_id] = generate_scores(models[0], sample, candidate_list, dataset.src_dict, use_cuda)
        gen_timer.stop(1)

    with open(args.results_path, 'wb') as handle:
        pickle.dump(scores, handle)

    logger.info('Predicted {} sentences in {:.1f}s ({:.2f} sentences/s)'.format(
        num_sentences, gen_timer.sum, num_sentences / gen_timer.sum))


def cli_main():
    # This script computes the similarity scores for the provided
    # candidates with respect to the input speech segments.
    # It follows the similarity method described as multi-word
    # in the weekly updates.
    parser = options.get_generation_parser()
    parser.add_argument("--list-candidates", type=str, required=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
