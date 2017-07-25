import argparse
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from progress_bar import progress_bar
from torch.autograd import Variable

import bleu
import data
import models
import utils
from meters import StopwatchMeter, TimeMeter


parser = argparse.ArgumentParser(description='Convolutional Sequence to Sequence Generation')
parser.add_argument('data', metavar='DIR',
                    help='path to data directory')
parser.add_argument('--path', metavar='FILE', default='./checkpoint_best.pt',
                    help='path to model file')

# dataset and data loading
parser.add_argument('--subset', default='test', metavar='SPLIT',
                    choices=['train', 'valid', 'test'],
                    help='data subset to generate (train, valid, test)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='batch size')

# generation configuration
parser.add_argument('--beam', default=5, type=int, metavar='N',
                    help='beam size')
parser.add_argument('--nbest', default=1, type=int, metavar='N',
                    help='number of hypotheses to output')
parser.add_argument('--max-len-a', default=0, type=int, metavar='N',
                    help=('generate sequence of maximum length ax + b, '
                          'where x is the source length'))
parser.add_argument('--max-len-b', default=200, type=int, metavar='N',
                    help=('generate sequence of maximum length ax + b, '
                          'where x is the source length'))
parser.add_argument('--no-early-stop', action='store_true',
                    help=('continue searching even after finalizing k=beam '
                          'hypotheses; this is more correct, but increases '
                          'generation time by 50%%'))
parser.add_argument('--unnormalized', action='store_true',
                    help='compare unnormalized hypothesis scores')

# misc
parser.add_argument('--cpu', action='store_true', help='generate on CPU')
parser.add_argument('--beamable-mm', action='store_true',
                    help='use BeamableMM in attention layers')
parser.add_argument('--no-progress-bar', action='store_true',
                    help='disable progress bar')

# model configuration
# TODO infer this from model file
parser.add_argument('--arch', '-a', default='fconv', metavar='ARCH',
                    choices=models.__all__,
                    help='model architecture ({})'.format(', '.join(models.__all__)))
parser.add_argument('--encoder-embed-dim', default=512, type=int, metavar='N',
                    help='encoder embedding dimension')
parser.add_argument('--encoder-layers', default='[(512, 3)] * 20', type=str, metavar='EXPR',
                    help='encoder layers [(dim, kernel_size), ...]')
parser.add_argument('--decoder-embed-dim', default=512, type=int, metavar='N',
                    help='decoder embedding dimension')
parser.add_argument('--decoder-layers', default='[(512, 3)] * 20', type=str, metavar='EXPR',
                    help='decoder layers [(dim, kernel_size), ...]')
parser.add_argument('--decoder-attention', default='True', type=str, metavar='EXPR',
                    help='decoder attention [True, ...]')
parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                    help='dropout probability')
parser.add_argument('--label-smoothing', default=0, type=float, metavar='D',
                    help='epsilon for label smoothing, 0 means no label smoothing')
parser.add_argument('--decoder-out-embed-dim', default=256, type=int, metavar='N',
                    help='decoder output embedding dimension')


def main():
    global args
    args = parser.parse_args()
    print(args)

    if args.no_progress_bar:
        progress_bar.enabled = False
    use_cuda = torch.cuda.is_available() and not args.cpu

    dataset = data.load(args.data)
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    print('| {} {} {} examples'.format(args.data, args.subset, len(dataset.splits[args.subset])))

    # TODO infer architecture from model file
    print('| model {}'.format(args.arch))
    model = utils.build_model(args, dataset)
    if use_cuda:
        model.cuda()

    # Load the model from the latest checkpoint
    epoch, _batch_offset = utils.load_checkpoint(args.path, model)

    # optimize model for generation
    model.make_generation_fast(args.beam, args.beamable_mm)

    def display_hypotheses(id, src, ref, hypos):
        print('S-{}\t{}'.format(id, to_sentence(dataset.src_dict, src)))
        print('T-{}\t{}'.format(id, to_sentence(dataset.dst_dict, ref)))
        for hypo in hypos:
            print('H-{}\t{}\t{}'.format(id, hypo['score'], to_sentence(dataset.dst_dict, hypo['tokens'])))

    # Initialize generator
    translator = SequenceGenerator(model, dataset.dst_dict, beam_size=args.beam,
                                   stop_early=(not args.no_early_stop),
                                   normalize_scores=(not args.unnormalized))
    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    scorer = bleu.Scorer(dataset.dst_dict.pad(), dataset.dst_dict.eos())
    itr = dataset.dataloader(args.subset, batch_size=args.batch_size)
    num_sentences = 0
    with progress_bar(itr, smoothing=0, leave=False) as t:
        wps_meter = TimeMeter()
        gen_timer = StopwatchMeter()
        translations = generate_batched_itr(
            translator, t, max_len_a=args.max_len_a, max_len_b=args.max_len_b,
            cuda_device=0 if use_cuda else None, timer=gen_timer)
        for id, src, ref, hypos in translations:
            wps_meter.update(src.size(0))
            scorer.add(ref.int().cpu(), hypos[0]['tokens'].int().cpu())
            t.set_postfix(wps='{:5d}'.format(round(wps_meter.avg)))
            display_hypotheses(id, src, ref, hypos[:min(len(hypos), args.nbest)])
            num_sentences += 1
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Generate {} with beam={}: BLEU4 = {:2.2f}'.format(args.subset, args.beam, scorer.score()))


def to_sentence(dict, tokens):
    if torch.is_tensor(tokens) and tokens.dim() == 2:
        sentences = [to_sentence(dict, token) for token in tokens]
        return '\n'.join(sentences)
    eos = dict.eos()
    return ' '.join([dict[i] for i in tokens if i != eos])


def expand_encoder_out(encoder_out, beam_size):
    res = []
    for tensor in encoder_out:
        res.append(
            # repeat beam_size times along second dimension
            tensor.repeat(1, beam_size, *[1 for i in range(tensor.dim()-2)]) \
                # then collapse into [bsz*beam, ...original dims...]
                .view(-1, *tensor.size()[1:])
        )
    return tuple(res)


def generate_batched_itr(translator, data_itr, max_len_a=0, max_len_b=200,
                         cuda_device=None, timer=None):
    '''Iterate over a batched dataset and yield individual translations.'''

    def lstrip_pad(tensor):
        return tensor[tensor.eq(translator.pad).sum():]

    for sample in data_itr:
        s = utils.prepare_sample(sample, volatile=True, cuda_device=cuda_device)
        input = s['net_input']
        srclen = input['src_tokens'].size(1)
        if timer is not None:
            timer.start()
        hypos = translator.generate(input['src_tokens'], input['src_positions'],
                                    maxlen=(max_len_a*srclen + max_len_b))
        if timer is not None:
            timer.stop(s['ntokens'])
        for i, id in enumerate(s['id']):
            src = input['src_tokens'].data[i, :]
            # remove padding from ref, which appears at the beginning
            ref = lstrip_pad(input['target'].data[i, :])
            yield id, src, ref, hypos[i]


class SequenceGenerator(object):
    def __init__(self, model, dst_dict, beam_size=1, minlen=1, maxlen=200,
                 stop_early=True, normalize_scores=True):
        '''Generates translations of a given source sentence.

        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        '''
        self.model = model
        self.dict = dst_dict
        self.pad = dst_dict.pad()
        self.eos = dst_dict.eos()
        self.vocab_size = len(dst_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.positions = torch.LongTensor(range(self.pad + 1, self.pad + maxlen + 2))
        self.decoder_context = model.decoder.context_size()
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores

    def generate(self, src_tokens, src_positions, beam_size=None, maxlen=None):
        with self.model.decoder.incremental_inference():
            return self._generate(src_tokens, src_positions, beam_size, maxlen)

    def _generate(self, src_tokens, src_positions, beam_size=None, maxlen=None):
        model = self.model
        model.eval()

        # start a fresh sequence
        model.decoder.clear_incremental_state()

        bsz = src_tokens.size(0)
        beam_size = beam_size if beam_size is not None else self.beam_size
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # compute the encoder output once and expand to beam size
        encoder_out = model.encoder(src_tokens, src_positions)
        encoder_out = expand_encoder_out(encoder_out, beam_size)

        # initialize buffers
        scores = encoder_out[0].data.new(bsz * beam_size).fill_(0)
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens[:, 0] = self.eos
        tokens_buf = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': float('Inf')} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz)*beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}
        def buffer(name, type_of=tokens):
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                bbsz = sent*beam_size
                best_unfinalized_score = scores[bbsz:bbsz+beam_size].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                scores: A vector of the same size as bbsz_idx containing scores
                    for each hypothesis
            """
            assert bbsz_idx.numel() == scores.numel()
            norm_scores = scores/(step+1) if self.normalize_scores else scores
            sents_seen = set()
            for idx, score in zip(bbsz_idx.cpu(), norm_scores.cpu()):
                sent = idx // beam_size
                sents_seen.add(sent)
                def get_hypo():
                    hypo = tokens[idx, 1:step+2].clone()
                    hypo[step] = self.eos
                    return {'tokens': hypo, 'score': score}
                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }
            # return number of hypotheses finished this step
            num_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        reorder_state = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder's internal state based on the prev choice of beams
            if reorder_state is not None:
                model.decoder.reorder_incremental_state(reorder_state)

            probs = self.decode(tokens[:, :step+1], encoder_out)
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
            else:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores.view(-1, 1))

            # take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            probs.view(bsz, -1).topk(cand_size, out=(cand_scores, cand_indices))
            torch.div(cand_indices, self.vocab_size, out=cand_beams)
            cand_indices.fmod_(self.vocab_size)

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                eos_bbsz_idx = buffer('eos_bbsz_idx')
                cand_bbsz_idx.masked_select(eos_mask, out=eos_bbsz_idx)
                if eos_bbsz_idx.numel() > 0:
                    eos_scores = buffer('eos_scores', type_of=scores)
                    cand_scores.masked_select(eos_mask, out=eos_scores)
                    num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add((eos_mask*cand_size).type_as(cand_offsets), cand_offsets,
                      out=active_mask)

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            active_mask.topk(beam_size, 1, largest=False, out=(_ignore, active_hypos))
            active_bbsz_idx = buffer('active_bbsz_idx')
            cand_bbsz_idx.gather(1, active_hypos, out=active_bbsz_idx)
            active_scores = cand_scores.gather(1, active_hypos,
                                               out=scores.view(bsz, beam_size))

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # finalize all active hypotheses once we hit maxlen
            # finalize_hypos will take care of adding the EOS markers
            if step == maxlen:
                num_remaining_sent -= finalize_hypos(step, active_bbsz_idx, active_scores)
                assert num_remaining_sent == 0
                break

            # copy tokens for active hypotheses
            torch.index_select(tokens[:, :step+1], dim=0, index=active_bbsz_idx,
                               out=tokens_buf[:, :step+1])
            cand_indices.gather(1, active_hypos,
                                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step+1])

            # swap buffers
            old_tokens = tokens
            tokens = tokens_buf
            tokens_buf = old_tokens

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(bsz):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def decode(self, tokens, encoder_out):
        length = tokens.size(1)

        # repeat the first length positions to fill batch
        positions = self.positions[:length].view(1, length)

        # wrap in Variables
        tokens = Variable(tokens, volatile=True)
        positions = Variable(positions, volatile=True)

        decoder_out = self.model.decoder(tokens, positions, encoder_out)
        probs = F.log_softmax(decoder_out[:, -1, :]).data
        return probs

    def cuda(self):
        self.model.cuda()
        self.positions = self.positions.cuda()
        return self


if __name__ == '__main__':
    main()
