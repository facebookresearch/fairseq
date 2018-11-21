import torch
import torch.nn as nn

from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.modules import (
    ElmoTokenEmbedder, MultiheadAttention,
    CharacterTokenEmbedder)
from . import (
    BaseFairseqModel, register_model, register_model_architecture,
)

from fairseq import options
from fairseq import utils


@register_model('finetuning_squad')
class FinetuningSquad(BaseFairseqModel):
    def __init__(self, args, language_model, eos_idx, pad_idx, unk_idx):
        super().__init__()

        self.language_model = language_model
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.last_dropout = nn.Dropout(args.last_dropout)

        self.ln = nn.LayerNorm(args.model_dim, elementwise_affine=False) if args.layer_norm else None

        self.start_proj = torch.nn.Linear(args.model_dim, 1, bias=True)
        self.end_proj = torch.nn.Linear(args.model_dim, 1, bias=True)

        if args.concat_sentences_mode == 'eos':
            mult = 3
        elif args.concat_sentences_mode == 'unk_only':
            mult = 2 + int(args.proj_unk)
        else:
            mult = 4 + int(args.proj_unk)

        self.imp_proj = torch.nn.Linear(args.model_dim * mult, 2, bias=True)
        self.proj_unk = args.proj_unk

        if isinstance(self.language_model.decoder.embed_tokens, CharacterTokenEmbedder):
            print('disabling training char convolutions')
            self.language_model.decoder.embed_tokens.disable_convolutional_grads()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.start_proj.weight, 0)
        torch.nn.init.constant_(self.start_proj.bias, 0)

        torch.nn.init.constant_(self.end_proj.weight, 0)
        torch.nn.init.constant_(self.end_proj.bias, 0)

        torch.nn.init.constant_(self.imp_proj.weight, 0)
        torch.nn.init.constant_(self.imp_proj.bias, 0)

    def forward(self, text, paragraph_mask):
        x, _ = self.language_model(text)
        if isinstance(x, list):
            x = x[0]

        if self.ln is not None:
            x = self.ln(x)

        idxs = text.eq(self.eos_idx)
        if self.proj_unk:
            idxs = idxs | text.eq(self.unk_idx)

        x = self.last_dropout(x)

        eos_emb = x[idxs].view(text.size(0), 1, -1)  # assume only 3 eoses per sample

        imp = self.imp_proj(eos_emb).squeeze(1)
        if paragraph_mask.any():
            paragraph_toks = x[paragraph_mask]
            start = x.new_full(paragraph_mask.shape, float('-inf'))
            end = start.clone()
            start[paragraph_mask] = self.start_proj(paragraph_toks).squeeze(-1)
            end[paragraph_mask] = self.end_proj(paragraph_toks).squeeze(-1)
        else:
            start = end = x.new_zeros(0)

        return imp, start, end

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--lm-path', metavar='PATH', help='path to elmo model')
        parser.add_argument('--model-dim', type=int, metavar='N', help='decoder input dimension')
        parser.add_argument('--last-dropout', type=float, metavar='D', help='dropout before projection')
        parser.add_argument('--model-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--relu-dropout', type=float, metavar='D', help='lm dropout')
        parser.add_argument('--proj-unk', action='store_true', help='if true, also includes unk emb in projection')
        parser.add_argument('--layer-norm', action='store_true', help='if true, does non affine layer norm before proj')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        dictionary = task.dictionary

        assert args.lm_path is not None

        task = LanguageModelingTask(args, dictionary, dictionary)
        models, _ = utils.load_ensemble_for_inference([args.lm_path], task, {
            'remove_head': True,
            'dropout': args.model_dropout,
            'attention_dropout': args.attention_dropout,
            'relu_dropout': args.relu_dropout,
        })
        assert len(models) == 1, 'ensembles are currently not supported for elmo embeddings'

        return FinetuningSquad(args, models[0], dictionary.eos(), dictionary.pad(), dictionary.unk())


@register_model_architecture('finetuning_squad', 'finetuning_squad')
def base_architecture(args):
    args.model_dim = getattr(args, 'model_dim', 1024)
    args.last_dropout = getattr(args, 'last_dropout', 0.1)
    args.model_dropout = getattr(args, 'model_dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.05)
    args.layer_norm = getattr(args, 'layer_norm', False)
    args.proj_unk = getattr(args, 'proj_unk', False)
