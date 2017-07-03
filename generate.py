import torch
import torch.nn.functional as F
from progress_bar import progress_bar
from torch.autograd import Variable

import bleu


def to_sentence(dict, tokens):
    if torch.is_tensor(tokens) and tokens.dim() == 2:
        sentences = [to_sentence(dict, token) for token in tokens]
        return '\n'.join(sentences)
    return ' '.join([dict[i] for i in tokens])


def expand_encoder_out(encoder_out, batch_size):
    res = []
    for tensor in encoder_out:
        res.append(tensor.expand(batch_size, *tensor.size()[1:]))
    return tuple(res)


class SequenceGenerator(object):
    def __init__(self, model, dst_dict, beam_size=1, minlen=1, maxlen=200):
        self.model = model
        self.dict = dst_dict
        self.pad = dst_dict.index('<pad>')
        self.eos = dst_dict.index('</s>')
        self.vocab_size = len(dst_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.positions = torch.LongTensor(range(self.pad + 1, self.pad + 1 + maxlen))
        self.decoder_context = model.decoder.context_size()

    def generate(self, src_tokens, src_positions):
        model = self.model
        model.eval()

        if self.positions.is_cuda:
            src_tokens = src_tokens.cuda()
            src_positions = src_positions.cuda()

        src_tokens = Variable(src_tokens, volatile=True)
        src_positions = Variable(src_positions, volatile=True)

        # compute the encoder output once
        encoder_out = model.encoder(src_tokens, src_positions)

        # start with beam size 1 and eos token
        tokens = torch.cuda.LongTensor(1, 1).fill_(self.eos)

        # initialize logit scores to zero
        scores = encoder_out[0].data.new(1).zero_()

        # list of completed sentences
        finalized = []

        for step in range(self.maxlen):
            probs = self.decode(tokens, encoder_out) + scores.view(-1, 1)

            if step < self.minlen:
                probs[:, self.eos] = float('-Inf')
            elif step == self.maxlen - 1:
                probs[:, self.eos] = float('Inf')

            # take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            next_scores, indices = probs.view(-1).topk(self.beam_size * 2)

            symbols = indices % self.vocab_size
            rows = indices / self.vocab_size

            idxs = []
            for j, (score, symbol, row) in enumerate(zip(next_scores.cpu(), symbols.cpu(), rows.cpu())):
                if len(idxs) == self.beam_size:
                    break
                if symbol != self.eos:
                    idxs.append(j)
                elif step > self.minlen and len(finalized) < self.beam_size:
                    finalized.append({
                        "tokens": tokens[row][1:].tolist() + [symbol],
                        "score": score,
                    })

            if len(finalized) >= self.beam_size:
                break

            idxs = torch.cuda.LongTensor(idxs)
            scores = next_scores[idxs]
            tokens = torch.cat([tokens[rows[idxs]], symbols[idxs]], dim=1)

        # sort by score descending
        finalized = sorted(finalized, key=lambda r: r['score'], reverse=True)

        return finalized

    def decode(self, tokens, encoder_out):
        beam, length = tokens.size()

        # repeat the first length positions to fill beam
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
            encoder_out[0].expand(beam, *encoder_out[0].size()[1:]),
            encoder_out[1].expand(beam, *encoder_out[1].size()[1:]),
        )

        decoder_out = self.model.decoder(tokens, positions, encoder_out)
        probs = F.log_softmax(decoder_out[:, -1, :]).data
        return probs

    def cuda(self):
        self.model.cuda()
        self.positions = self.positions.cuda()
        return self


def generate(model, dataset):
    model.eval()

    pad = dataset.dst_dict.index('<pad>')
    eos = dataset.dst_dict.index('</s>')

    itr = dataset.dataloader('test', batch_size=1)
    translator = SequenceGenerator(model, dataset.dst_dict, beam_size=20, maxlen=200)
    translator.cuda()

    scorer = bleu.Scorer(pad, eos)
    with progress_bar(itr, smoothing=0, leave=False) as t:
        for sample in t:
            hypotheses = translator.generate(sample['src_tokens'], sample['src_positions'])

            ground_truth = sample['target'].int()
            pred = torch.IntTensor(hypotheses[0]['tokens'])

            scorer.add(ground_truth, pred)
            t.set_postfix(bleu='{:2.2f}'.format(scorer.score()))

    return scorer


def main():
    import models
    import data

    global model, dataset, translator

    dataset = data.load('iwslt14_de-en_bpe20k')
    model = models.fconv_iwslt_de_en(dataset, 0.2)
    model.cuda()

    model.load_state_dict(torch.load('checkpoint.pt')['model'])

    translator = SequenceGenerator(model, dataset.dst_dict, beam_size=20, maxlen=200)
    translator.cuda()

    global sample, src_tokens, src_positions
    sample = dataset.splits['test'][4214]

    src_tokens = sample['src_tokens'].view(1, -1).cuda()
    src_positions = sample['src_positions'].view(1, -1).cuda()
    hypos = translator.generate(src_tokens, src_positions)

    print(to_sentence(dataset.src_dict, sample['src_tokens']))
    print(to_sentence(dataset.dst_dict, sample['target']))
    if len(hypos) > 0:
        print(to_sentence(dataset.dst_dict, hypos[0]['tokens']))


if __name__ == '__main__':
    main()
