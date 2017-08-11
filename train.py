import collections
import os
import torch
import math

from fairseq import bleu, data, options, utils
from fairseq.meters import AverageMeter, TimeMeter
from fairseq.multiprocessing_trainer import MultiprocessingTrainer
from fairseq.progress_bar import progress_bar
from fairseq.sequence_generator import SequenceGenerator


def main():
    parser = options.get_parser('Trainer')
    dataset_args = options.add_dataset_args(parser)
    dataset_args.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                              help='maximum number of tokens in a batch')
    dataset_args.add_argument('--train-subset', default='train', metavar='SPLIT',
                              choices=['train', 'valid', 'test'],
                              help='data subset to use for training (train, valid, test)')
    dataset_args.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                              choices=['train', 'valid', 'test'],
                              help='data subset to use for validation (train, valid, test)')
    dataset_args.add_argument('--test-subset', default='test', metavar='SPLIT',
                              choices=['train', 'valid', 'test'],
                              help='data subset to use for testing (train, valid, test)')
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    options.add_model_args(parser)
    args = parser.parse_args()
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False
        progress_bar.print_interval = args.log_interval

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.manual_seed(args.seed)

    dataset = data.load(args.data, args.source_lang, args.target_lang)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in ['train', 'valid', 'test']:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    num_gpus = torch.cuda.device_count()

    print('| using {} GPUs (with max tokens per GPU = {})'.format(num_gpus, args.max_tokens))

    print('| model {}'.format(args.arch))
    model = utils.build_model(args, dataset)

    # Start multiprocessing
    trainer = MultiprocessingTrainer(args, model)

    # Load the latest checkpoint if one is available
    epoch, batch_offset = trainer.load_checkpoint(
        os.path.join(args.save_dir, args.restore_file))

    # Train until the learning rate gets too small
    val_loss = None
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    while lr > args.min_lr and epoch <= max_epoch:
        # train for one epoch
        train(epoch, batch_offset, trainer, dataset, num_gpus)

        # evaluate on validate set
        val_loss, lr = validate(epoch, trainer, dataset, num_gpus)

        # save checkpoint
        trainer.save_checkpoint(args.save_dir, epoch, 0, val_loss)

        epoch += 1
        batch_offset = 0

    # Generate on test set and compute BLEU score
    for beam in [1, 5, 10, 20]:
        scorer = score_test(epoch, trainer.get_model(), dataset, beam, 0 if num_gpus > 0 else None)
        print('| Test with beam={}: BLEU4 = {:2.2f}'.format(beam, scorer.score()))

    # Stop multiprocessing
    trainer.stop()


def train(epoch, batch_offset, trainer, dataset, num_gpus):
    """Train the model for one epoch"""

    itr = dataset.dataloader(args.train_subset, num_workers=args.workers,
                             max_tokens=args.max_tokens, seed=(args.seed, epoch))
    loss_meter = AverageMeter()
    bsz_meter = AverageMeter()  # sentences per batch
    wpb_meter = AverageMeter()  # words per batch
    wps_meter = TimeMeter()     # words per second

    desc = '| epoch {}'.format(epoch)
    lr = trainer.get_lr()
    with progress_bar(itr, desc, leave=False) as t:
        for i, sample in skip_group_enumerator(t, num_gpus, batch_offset):
            loss = trainer.train_step(sample)

            ntokens = sum(s['ntokens'] for s in sample)
            src_size = sum(s['src_tokens'].size(0) for s in sample)
            loss_meter.update(loss, ntokens)
            bsz_meter.update(src_size)
            wpb_meter.update(ntokens)
            wps_meter.update(ntokens)

            t.set_postfix(collections.OrderedDict([
                ('loss', '{:.2f} ({:.2f})'.format(loss, loss_meter.avg)),
                ('wps', '{:5d}'.format(round(wps_meter.avg))),
                ('wpb', '{:5d}'.format(round(wpb_meter.avg))),
                ('bsz', '{:5d}'.format(round(bsz_meter.avg))),
                ('lr', lr),
            ]))

            if i == 0:
                # ignore the first mini-batch in words-per-second calculation
                wps_meter.reset()
            if args.save_interval > 0 and (i + 1) % args.save_interval == 0:
                trainer.save_checkpoint(args.save_dir, epoch, i + 1)

        fmt = '| epoch {:03d} | train loss {:2.2f} | train ppl {:3.2f}'
        fmt += ' | s/checkpoint {:7d} | words/s {:6d} | words/batch {:6d}'
        fmt += ' | bsz {:5d} | lr {:0.6f}'
        t.write(fmt.format(epoch, loss_meter.avg, math.pow(2, loss_meter.avg),
                           round(wps_meter.elapsed_time),
                           round(wps_meter.avg),
                           round(wpb_meter.avg),
                           round(bsz_meter.avg),
                           lr))


def skip_group_enumerator(it, ngpus, offset=0):
    res = []
    idx = 0
    for i, sample in enumerate(it):
        if i < offset:
            continue
        res.append(sample)
        if len(res) >= ngpus:
            yield (i, res)
            res = []
            idx = i + 1
    if len(res) > 0:
        yield (idx, res)


def validate(epoch, trainer, dataset, ngpus):
    """Evaluate the model on the validation set and return the average loss"""

    itr = dataset.dataloader(args.valid_subset, batch_size=None, max_tokens=args.max_tokens)
    loss_meter = AverageMeter()

    desc = '| val {}'.format(epoch)
    with progress_bar(itr, desc, leave=False) as t:
        for _, sample in skip_group_enumerator(t, ngpus):
            ntokens = sum(s['ntokens'] for s in sample)
            loss = trainer.valid_step(sample)
            loss_meter.update(loss, ntokens)
            t.set_postfix(loss='{:.2f}'.format(loss_meter.avg))

        val_loss = loss_meter.avg
        t.write('| epoch {:03d} | val loss {:2.2f} | val ppl {:3.2f}'
                .format(epoch, val_loss, math.pow(2, val_loss)))

    # update and return the learning rate
    return val_loss, trainer.lr_step(val_loss, epoch)


def score_test(epoch, model, dataset, beam, cuda_device=None):
    """Evaluate the model on the test set and print the BLEU score"""
    translator = SequenceGenerator(model, dataset.dst_dict, beam_size=beam)
    if torch.cuda.is_available():
        translator.cuda()

    scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos())
    itr = dataset.dataloader(args.test_subset, batch_size=4)
    for id, src, ref, hypos in generate.generate_batched_itr(translator, itr, cuda_device=cuda_device):
        scorer.add(ref.int().cpu(), hypos[0]['tokens'].int().cpu())
    return scorer


if __name__ == '__main__':
    main()
