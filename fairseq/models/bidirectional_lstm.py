import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    FairseqDecoder, FairseqLanguageModel, register_model, register_model_architecture,
)

from fairseq import options
from fairseq import utils
from fairseq.modules import (
    AdaptiveSoftmax, LstmCellWithProjection, CharacterTokenEmbedder
)

@register_model('bi_lstm_lm')
class LSTMLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', default=None, type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--decoder-memory-dim', type=int, metavar='N',
                            help='decoder memory cell dim')
        parser.add_argument('--memory-clip-value', type=float, metavar='D',
                            help='memory clip value')
        parser.add_argument('--state-clip-value', type=float, metavar='D',
                            help='state clip value')
        parser.add_argument('--decoder-bidirectional', type=str, metavar='BOOL',
                            help='decoder directional')
        parser.add_argument('--residual-connection', type=str, metavar='BOOL',
                            help='residual connections between lstm layers')

        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--exclude-self-target', action='store_true',
                            help='exclude self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_lm_architecture(args)
        targets = ['self'] if not args.exclude_self_target else []
        if args.future_target:
            targets.append('future')
        if args.past_target:
            targets.append('past')

        for ds in task.datasets.values():
            ds.set_targets(targets)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        decoder = BiLSTMDecoder(
            dictionary=task.target_dictionary,
            targets=targets,
            character_embeddings = args.character_embeddings,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            adaptive_softmax_cutoff=
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
            bidirectional=args.decoder_bidirectional,
            memory_dim=args.decoder_memory_dim,
            memory_clip_value=args.memory_clip_value,
            state_clip_value=args.state_clip_value,
            residual=args.residual_connection,
            character_filters=args.character_filters,
            character_embedding_dim = args.character_embedding_dim,
            char_embedder_highway_layers=args.char_embedder_highway_layers
        )
        return cls(decoder)

class BiLSTMDecoder(FairseqDecoder):
    """BiLSTM decoder."""

    def __init__(
            self, dictionary, targets, character_embeddings, embed_dim=512, hidden_size=512, out_embed_dim=512,
            num_layers=2, dropout_in=0.1, dropout_out=0.1,
            encoder_output_units=512, adaptive_softmax_cutoff=None,
            bidirectional=True, memory_dim=4096, memory_clip_value=3.0, state_clip_value=3.0,
            residual=True, character_filters=None,character_embedding_dim=16, char_embedder_highway_layers =0, 
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.memory_dim = memory_dim
        self.memory_clip_value = memory_clip_value
        self.state_clip_value = state_clip_value
        self.residual = residual
        self.targets = targets
        self.num_layers = num_layers

        if character_embeddings:
            self.embed_tokens = CharacterTokenEmbedder(dictionary, eval(character_filters),
                                                  character_embedding_dim,
                                                  embed_dim,
                                                  char_embedder_highway_layers
                                                  )
        else:
            self.embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        
        
        assert num_layers > 0

        for layer_index in range(self.num_layers):
            forward_layer = LstmCellWithProjection(input_size=embed_dim if layer_index == 0 else hidden_size,
                                                   hidden_size=hidden_size,
                                                   cell_size=memory_dim,
                                                   go_forward=True,
                                                   recurrent_dropout_probability=dropout_out,
                                                   memory_cell_clip_value=memory_clip_value,
                                                   state_projection_clip_value=state_clip_value,
                                                   is_training=self.training)
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            
            if bidirectional:

                backward_layer = LstmCellWithProjection(input_size=embed_dim if layer_index == 0 else hidden_size,
                                                        hidden_size=hidden_size,
                                                        cell_size=memory_dim,
                                                        go_forward=False,
                                                        recurrent_dropout_probability=dropout_out,
                                                        memory_cell_clip_value=memory_clip_value,
                                                        state_projection_clip_value=state_clip_value,
                                                        is_training=self.training)
                self.add_module('backward_layer_{}'.format(layer_index), backward_layer)

        self.adaptive_softmax = self.additional_fc = self.fc_out = None

        if adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(len(dictionary), out_embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        else:
            if hidden_size != out_embed_dim:
                self.additional_fc = Linear(hidden_size, out_embed_dim, dropout=dropout_out)
            self.fc_out = Linear(out_embed_dim, len(dictionary), dropout=dropout_out)
            self.adaptive_softmax = None

    def forward(self, inputs, states=None):
        """ Forward pass for the bidirectional transformer
        Args:
            - source tokens: B x T matrix representing sentences
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in softmax afterwards
                - a dictionary of additional data, where 'attn' contains the attention over the final
                  states (concatenated from forward and backward towers) and 'inner_states' is a list
                  of internal model states used to compute the predictions (for example to use in ELMO).
                  The first element is the token embeddings (with the positional embeddings added).
                  The next n elements are tuples of the hidden states for the forward and backward towers.
                  The last element is the output of the final full layer on top of the towers and would be
                  equivalent to the logits if adaptive softmax is used.
                  NOTE: unlike the logits, the format for all hidden states is T x B x C
        """
        bsz, seqlen = inputs.size()
        
        # embed tokens
        x = self.embed_tokens(inputs)
        
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        
        if not states:
            states = [[None for j in range(self.num_layers)] for _ in range(self.directions)]

        inputs = [x, x]
        res = []
        states_res =[]
        inner_states = [x]

        for idx, input in enumerate(inputs):
           
            output_sequence = input
            state = states[idx]
           
            if state[0]:
               
                state = [[Variable(s, requires_grad = False) for s in state[layer_idx]] for layer_idx in range(self.num_layers)]
           
            for layer_index in range(self.num_layers):
               
                cache = output_sequence
               
                if idx == 0:
                    layer = getattr(self, 'forward_layer_{}'.format(layer_index))
                else:
                    layer = getattr(self, 'backward_layer_{}'.format(layer_index))
               
                output_sequence, state[layer_index] = layer(output_sequence, state[layer_index])
               
                if layer_index > 0 and self.residual:
                    output_sequence = output_sequence + cache 
                inner_states.append(output_sequence)

            res.append(output_sequence)
            states_res.append(state)
        
        if self.adaptive_softmax is None:
            if self.additional_fc:
                res = [self.additional_fc(x) for x in res]
            if self.fc_out:
                res = [self.fc_out(x) for x in res]
        
        return res, {'states' : states_res, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return 1e15


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('bi_lstm_lm', 'bi_lstm_lm')
def base_lm_architecture(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '0')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.1)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.1)
    args.decoder_memory_dim = getattr(args, 'decoder_memory_dim', 4096)
    args.memory_clip_value = getattr(args, 'memory_clip_value', 3.0)
    args.state_clip_value = getattr(args, 'state_clip_value', 3.0)
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', True)
    args.residual_connection = getattr(args, 'residual_connection', True)
    args.character_embeddings = getattr(args, 'character_embeddings', False)
    args.character_filters = getattr(args, 'character_filters',
                                     '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]')
    args.character_embedding_dim = getattr(args, 'character_embedding_dim', 128)
    args.char_embedder_highway_layers = getattr(args, 'char_embedder_highway_layers', 2)

    args.exclude_self_target = getattr(args, 'exclude_self_target', False)
    args.future_target = getattr(args, 'future_target', False)
    args.past_target = getattr(args, 'past_target', False)
