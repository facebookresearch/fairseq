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
from average_meter import TimeMeter


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
    with progress_bar(itr, smoothing=0, leave=False) as t:
        wps_meter = TimeMeter()
        translations = generate_batched_itr(
            translator, t, max_len_a=args.max_len_a, max_len_b=args.max_len_b,
            cuda_device=0 if use_cuda else None)
        for id, src, ref, hypos in translations:
            wps_meter.update(src.size(0))
            scorer.add(ref.int().cpu(), hypos[0]['tokens'].int().cpu())
            t.set_postfix(wps='{:5d}'.format(round(wps_meter.avg)))
            display_hypotheses(id, src, ref, hypos[:min(len(hypos), args.nbest)])
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
                         cuda_device=None):
    '''Iterate over a batched dataset and yield individual translations.'''

    def lstrip_pad(tensor):
        return tensor[tensor.eq(translator.pad).sum():]

    for sample in data_itr:
        s = utils.prepare_sample(sample, volatile=True, cuda_device=cuda_device)
        input = s['net_input']
        srclen = input['src_tokens'].size(1)
        hypos = translator.generate(
            input['src_tokens'], input['src_positions'],
            maxlen=(max_len_a*srclen + max_len_b)
        )
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

    def generate(self, src_tokens, src_positions, maxlen=None):
        model = self.model
        model.eval()

        # compute the encoder output once and expand to beam size
        encoder_out = model.encoder(src_tokens, src_positions)
        beam_encoder_out = expand_encoder_out(encoder_out, self.beam_size)

        bsz = src_tokens.size(0)
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # initialize buffers
        scores = encoder_out[0].data.new(bsz * self.beam_size, 1).fill_(0)
        tokens = src_tokens.data.new(bsz * self.beam_size, maxlen + 2).fill_(self.pad)
        tokens[:, 0] = self.eos
        tokens_buf = src_tokens.data.new(bsz * self.beam_size, maxlen + 2).fill_(self.pad)
        row_idx_to_copy = src_tokens.data.new(bsz * self.beam_size)

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': float('Inf')} for i in range(bsz)]
        num_remaining_sentences = bsz

        def finalize(sent, hypo, score):
            '''Finalize a given hypothesis, while keeping the total number
            of finalized hypotheses <= beam_size.'''
            sz = hypo.size(0)  # includes EOS
            norm_score = score / sz if self.normalize_scores else score
            h_new = {
                'tokens': hypo,
                'score': norm_score,
            }
            if len(finalized[sent]) < self.beam_size:
                finalized[sent].append(h_new)
                if norm_score < worst_finalized[sent]['score']:
                    worst_finalized[sent] = {
                        'score': norm_score,
                        'idx': len(finalized[sent]) - 1,
                    }
            elif norm_score > worst_finalized[sent]['score']:
                # replace worst hypo for this sentence with new/better one
                worst_idx = worst_finalized[sent]['idx']
                finalized[sent][worst_idx] = h_new

                # find new worst finalized hypo for this sentence
                idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                worst_finalized[sent] = {
                    'score': s['score'],
                    'idx': idx,
                }

        def is_finished(sent):
            '''Check whether we've finished generation for a given sentence,
            by comparing the worst score among finalized hypotheses to the
            best possible score among unfinalized hypotheses.'''
            if len(finalized[sent]) >= self.beam_size:
                if self.stop_early:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                bbsz = sent*self.beam_size
                best_unfinalized_score = scores[bbsz:bbsz+self.beam_size].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        for step in range(maxlen + 1):  # one extra step for EOS marker
            if step == 0:
                probs = self.decode(tokens[::self.beam_size, :1], encoder_out)
            else:
                probs = self.decode(tokens[:, :step+1], beam_encoder_out).add_(scores)

            if step < self.minlen:
                probs[:, self.eos] = float('-Inf')

            # take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            next_scores, indices = probs.view(bsz, -1).topk(self.beam_size * 2)
            beams = indices / self.vocab_size
            indices = indices % self.vocab_size

            # keep track of active/unfinished hypotheses for next step
            num_active_hypos = [0 for i in range(bsz)]
            row_idx_to_copy.copy_(torch.arange(0, bsz * self.beam_size))

            for sent in range(indices.size(0)):  # bsz
                if finished[sent]:  # minor optimization
                    continue

                for i in range(indices.size(1)):  # beam_size*2
                    beam = beams[sent, i]
                    token = indices[sent, i]
                    score = next_scores[sent, i]
                    bbsz_idx = sent*self.beam_size + beam

                    # finalize hypotheses that generate eos,
                    # and finalize all hypotheses once we hit maxlen
                    if token == self.eos or step == maxlen:
                        if step < self.minlen:
                            continue
                        hypo = tokens.new(step + 1)
                        hypo[:step] = tokens[bbsz_idx][1:step+1]
                        hypo[step] = self.eos
                        finalize(sent, hypo, score)

                    # pick top-scoring unfinished hypotheses for next step
                    else:
                        if num_active_hypos[sent] >= self.beam_size:
                            # we already have enough hypos, skip
                            continue
                        new_bbsz_idx = sent*self.beam_size + num_active_hypos[sent]
                        # instead of copying tokens from previous steps now,
                        # buffer the row indices and copy them with index_select
                        row_idx_to_copy[new_bbsz_idx] = bbsz_idx
                        tokens_buf[new_bbsz_idx, step+1] = token
                        scores[new_bbsz_idx] = score
                        num_active_hypos[sent] += 1

                # check termination conditions for this sentence
                if is_finished(sent):
                    finished[sent] = True
                    num_remaining_sentences -= 1
                else:
                    assert num_active_hypos[sent] == self.beam_size

            # either we're done or we need to copy tokens for the next step
            if num_remaining_sentences == 0:
                break
            else:
                # copy tokens from previous steps
                torch.index_select(tokens[:, :step+1], dim=0, index=row_idx_to_copy,
                                   out=tokens_buf[:, :step+1])
                # swap buffers
                old_tokens = tokens
                tokens = tokens_buf
                tokens_buf = old_tokens

        # sort by score descending
        for sent in range(bsz):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def decode(self, tokens, encoder_out):
        bbsz, length = tokens.size()

        # repeat the first length positions to fill batch
        positions = self.positions[:length].view(1, length).expand_as(tokens)

        # only feed in as many tokens as the decoder needs based on the
        # convolution kernel sizes
        if length > self.decoder_context:
            tokens = tokens[:, -self.decoder_context:]
            positions = positions[:, -self.decoder_context:]

        # wrap in Variables
        tokens = Variable(tokens, volatile=True)
        positions = Variable(positions, volatile=True)

        encoder_out = (
            encoder_out[0].expand(bbsz, *encoder_out[0].size()[1:]),
            encoder_out[1].expand(bbsz, *encoder_out[1].size()[1:]),
        )

        decoder_out = self.model.decoder(tokens, positions, encoder_out)
        probs = F.log_softmax(decoder_out[:, -1, :]).data
        return probs

    def cuda(self):
        self.model.cuda()
        self.positions = self.positions.cuda()
        return self


if __name__ == '__main__':
    main()
