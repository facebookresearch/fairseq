
import logging
import os
import pickle
import sys

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter


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


def generate_sim_matrix(model, sample):
    with torch.no_grad():
        encoder_input = {
            k: v
            for k, v in sample['net_input'].items()
            if k != "prev_output_tokens"
        }
        encoder_outs = model.encoder.forward(**encoder_input)
        return similarity_matrix(
            teacher_states=encoder_outs[0]["encoder_states"],
            student_states=encoder_outs[1]["encoder_states"],
            teacher_masking=encoder_outs[0]["encoder_padding_mask"],
            student_masking=encoder_outs[1]["encoder_padding_mask"],
        )


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
    logger = logging.getLogger('fairseq_cli.s2t_analysis')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

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

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    num_sentences = 0
    matrices = {}
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        gen_timer.start()
        batch_matrices = generate_sim_matrix(models[0], sample).cpu()
        gen_timer.stop(1)
        ids = sample['id']
        for i in range(len(ids)):
            matrices[ids[i].item()] = batch_matrices[i]

    with open(args.results_path, 'wb') as handle:
        pickle.dump(matrices, handle)

    logger.info('Predicted {} sentences in {:.1f}s ({:.2f} sentences/s)'.format(
        num_sentences, gen_timer.sum, num_sentences / gen_timer.sum))


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
