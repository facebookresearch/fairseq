# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel


class LSTMModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(self, dictionary, embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1):
        super().__init__()
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.layers = nn.ModuleList([
            LSTMCell(embed_dim, embed_dim)
            for layer in range(num_layers)
        ])

    def forward(self, src_tokens):
        bsz, seqlen = src_tokens.size()
        num_layers = len(self.layers)

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        final_hiddens, final_cells = [], []
        outs = [x[j] for j in range(seqlen)]
        for i, rnn in enumerate(self.layers):
            hidden = Variable(x.data.new(bsz, embed_dim).zero_())
            cell = Variable(x.data.new(bsz, embed_dim).zero_())
            for j in range(seqlen):
                # recurrent cell
                hidden, cell = rnn(outs[j], (hidden, cell))

                # store the most recent hidden state in outs, either to be used
                # as the input for the next layer, or as the final output
                outs[j] = F.dropout(hidden, p=self.dropout_out, training=self.training)

            # save the final hidden and cell states for every layer
            final_hiddens.append(hidden)
            final_cells.append(cell)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        final_hiddens = torch.cat(final_hiddens, dim=0).view(num_layers, bsz, embed_dim)
        final_cells = torch.cat(final_cells, dim=0).view(num_layers, bsz, embed_dim)

        return x, final_hiddens, final_cells

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2*output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, attention=True):
        super().__init__()
        self.dictionary = dictionary
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(encoder_embed_dim, embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, input_tokens, encoder_out):
        bsz, seqlen = input_tokens.size()
        num_layers = len(self.layers)

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(input_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        prev_hiddens = self.get_incremental_state('prev_hiddens')
        if not prev_hiddens:
            # first time step, initialize previous states
            prev_hiddens, prev_cells = self._init_prev_states(input_tokens, encoder_out)
            input_feed = Variable(x.data.new(bsz, embed_dim).zero_())
        else:
            # previous states are cached
            prev_cells = self.get_incremental_state('prev_cells')
            input_feed = self.get_incremental_state('input_feed')

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        self.set_incremental_state('prev_hiddens', prev_hiddens)
        self.set_incremental_state('prev_cells', prev_cells)
        self.set_incremental_state('input_feed', input_feed)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        x = self.fc_out(x)

        return x, attn_scores

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(new_order)
        new_order = Variable(new_order)

        def reorder_state(key):
            old = self.get_incremental_state(key)
            if isinstance(old, list):
                new = [old_i.index_select(0, new_order) for old_i in old]
            else:
                new = old.index_select(0, new_order)
            self.set_incremental_state(key, new)

        reorder_state('prev_hiddens')
        reorder_state('prev_cells')
        reorder_state('input_feed')

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def _init_prev_states(self, input_tokens, encoder_out):
        _, encoder_hiddens, encoder_cells = encoder_out
        bsz = input_tokens.size(0)
        num_layers = len(self.layers)
        embed_dim = encoder_hiddens.size(2)

        prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
        prev_cells = [encoder_cells[i] for i in range(num_layers)]
        return prev_hiddens, prev_cells


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_dim, hidden_dim, **kwargs):
    m = nn.LSTMCell(input_dim, hidden_dim, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def get_archs():
    return [
        'lstm', 'lstm_wiseman_iwslt_de_en', 'lstm_luong_wmt_en_de',
    ]


def _check_arch(args):
    """Check that the specified architecture is valid and not ambiguous."""
    if args.arch not in get_archs():
        raise ValueError('Unknown LSTM model architecture: {}'.format(args.arch))
    if args.arch != 'lstm':
        # check that architecture is not ambiguous
        for a in ['encoder_embed_dim', 'encoder_layers', 'decoder_embed_dim', 'decoder_layers',
                  'decoder_out_embed_dim']:
            if hasattr(args, a):
                raise ValueError('--{} cannot be combined with --arch={}'.format(a, args.arch))


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'lstm_wiseman_iwslt_de_en':
        args.encoder_embed_dim = 256
        args.encoder_layers = 1
        args.encoder_dropout_in = 0
        args.encoder_dropout_out = 0
        args.decoder_embed_dim = 256
        args.decoder_layers = 1
        args.decoder_out_embed_dim = 256
        args.decoder_attention = True
        args.decoder_dropout_in = 0
    elif args.arch == 'lstm_luong_wmt_en_de':
        args.encoder_embed_dim = 1000
        args.encoder_layers = 4
        args.encoder_dropout_out = 0
        args.decoder_embed_dim = 1000
        args.decoder_layers = 4
        args.decoder_out_embed_dim = 1000
        args.decoder_attention = True
        args.decoder_dropout_out = 0
    else:
        assert args.arch == 'lstm'

    # default architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', True)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    return args


def build_model(args, src_dict, dst_dict):
    encoder = LSTMEncoder(
        src_dict,
        embed_dim=args.encoder_embed_dim,
        num_layers=int(args.encoder_layers),
        dropout_in=args.encoder_dropout_in,
        dropout_out=args.encoder_dropout_out,
    )
    decoder = LSTMDecoder(
        dst_dict,
        encoder_embed_dim=args.encoder_embed_dim,
        embed_dim=args.decoder_embed_dim,
        out_embed_dim=args.decoder_out_embed_dim,
        num_layers=int(args.decoder_layers),
        attention=bool(args.decoder_attention),
        dropout_in=args.decoder_dropout_in,
        dropout_out=args.decoder_dropout_out,
    )
    return LSTMModel(encoder, decoder)
