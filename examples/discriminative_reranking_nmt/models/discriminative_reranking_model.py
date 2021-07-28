from dataclasses import dataclass, field
import os

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)

from fairseq.models.roberta.model import RobertaClassificationHead

from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    TransformerSentenceEncoderLayer,
)


ACTIVATION_FN_CHOICES = ChoiceEnum(utils.get_available_activation_fns())
JOINT_CLASSIFICATION_CHOICES = ChoiceEnum(["none", "sent"])
SENTENCE_REP_CHOICES = ChoiceEnum(["head", "meanpool", "maxpool"])


def update_init_roberta_model_state(state):
    """
   update the state_dict of a Roberta model for initializing
   weights of the BertRanker
   """
    for k in list(state.keys()):
        if ".lm_head." in k or "version" in k:
            del state[k]
            continue
        # remove 'encoder/decoder.sentence_encoder.' from the key
        assert k.startswith("encoder.sentence_encoder.") or k.startswith(
            "decoder.sentence_encoder."
        ), f"Cannot recognize parameter name {k}"
        if "layernorm_embedding" in k:
            new_k = k.replace(".layernorm_embedding.", ".emb_layer_norm.")
            state[new_k[25:]] = state[k]
        else:
            state[k[25:]] = state[k]
        del state[k]


class BaseRanker(nn.Module):
    def __init__(self, args, task):
        super().__init__()

        self.separator_token = task.dictionary.eos()
        self.padding_idx = task.dictionary.pad()

    def forward(self, src_tokens):
        raise NotImplementedError

    def get_segment_labels(self, src_tokens):
        segment_boundary = (src_tokens == self.separator_token).long()
        segment_labels = (
            segment_boundary.cumsum(dim=1)
            - segment_boundary
            - (src_tokens == self.padding_idx).long()
        )

        return segment_labels

    def get_positions(self, src_tokens, segment_labels):
        segment_positions = (
            torch.arange(src_tokens.shape[1])
            .to(src_tokens.device)
            .repeat(src_tokens.shape[0], 1)
        )
        segment_boundary = (src_tokens == self.separator_token).long()
        _, col_idx = (segment_positions * segment_boundary).nonzero(as_tuple=True)
        col_idx = torch.cat([torch.zeros(1).type_as(col_idx), col_idx])
        offset = torch.cat(
            [
                torch.zeros(1).type_as(segment_boundary),
                segment_boundary.sum(dim=1).cumsum(dim=0)[:-1],
            ]
        )
        segment_positions -= col_idx[segment_labels + offset.unsqueeze(1)] * (
            segment_labels != 0
        )

        padding_mask = src_tokens.ne(self.padding_idx)
        segment_positions = (segment_positions + 1) * padding_mask.type_as(
            segment_positions
        ) + self.padding_idx

        return segment_positions


class BertRanker(BaseRanker):
    def __init__(self, args, task):
        super(BertRanker, self).__init__(args, task)

        init_model = getattr(args, "pretrained_model", "")
        self.joint_layers = nn.ModuleList()
        if os.path.isfile(init_model):
            print(f"initialize weight from {init_model}")

            from fairseq import hub_utils

            x = hub_utils.from_pretrained(
                os.path.dirname(init_model),
                checkpoint_file=os.path.basename(init_model),
            )

            in_state_dict = x["models"][0].state_dict()
            init_args = x["args"].model

            num_positional_emb = init_args.max_positions + task.dictionary.pad() + 1

            # follow the setup in roberta
            self.model = TransformerSentenceEncoder(
                padding_idx=task.dictionary.pad(),
                vocab_size=len(task.dictionary),
                num_encoder_layers=getattr(
                    args, "encoder_layers", init_args.encoder_layers
                ),
                embedding_dim=init_args.encoder_embed_dim,
                ffn_embedding_dim=init_args.encoder_ffn_embed_dim,
                num_attention_heads=init_args.encoder_attention_heads,
                dropout=init_args.dropout,
                attention_dropout=init_args.attention_dropout,
                activation_dropout=init_args.activation_dropout,
                num_segments=2,  # add language embeddings
                max_seq_len=num_positional_emb,
                offset_positions_by_padding=False,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn=init_args.activation_fn,
                freeze_embeddings=args.freeze_embeddings,
                n_trans_layers_to_freeze=args.n_trans_layers_to_freeze,
            )

            # still need to learn segment embeddings as we added a second language embedding
            if args.freeze_embeddings:
                for p in self.model.segment_embeddings.parameters():
                    p.requires_grad = False

            update_init_roberta_model_state(in_state_dict)
            print("loading weights from the pretrained model")
            self.model.load_state_dict(
                in_state_dict, strict=False
            )  # ignore mismatch in language embeddings

            ffn_embedding_dim = init_args.encoder_ffn_embed_dim
            num_attention_heads = init_args.encoder_attention_heads
            dropout = init_args.dropout
            attention_dropout = init_args.attention_dropout
            activation_dropout = init_args.activation_dropout
            activation_fn = init_args.activation_fn

            classifier_embed_dim = getattr(
                args, "embed_dim", init_args.encoder_embed_dim
            )
            if classifier_embed_dim != init_args.encoder_embed_dim:
                self.transform_layer = nn.Linear(
                    init_args.encoder_embed_dim, classifier_embed_dim
                )
        else:
            self.model = TransformerSentenceEncoder(
                padding_idx=task.dictionary.pad(),
                vocab_size=len(task.dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.embed_dim,
                ffn_embedding_dim=args.ffn_embed_dim,
                num_attention_heads=args.attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                max_seq_len=task.max_positions()
                if task.max_positions()
                else args.tokens_per_sample,
                num_segments=2,
                offset_positions_by_padding=False,
                encoder_normalize_before=args.encoder_normalize_before,
                apply_bert_init=args.apply_bert_init,
                activation_fn=args.activation_fn,
            )

            classifier_embed_dim = args.embed_dim
            ffn_embedding_dim = args.ffn_embed_dim
            num_attention_heads = args.attention_heads
            dropout = args.dropout
            attention_dropout = args.attention_dropout
            activation_dropout = args.activation_dropout
            activation_fn = args.activation_fn

        self.joint_classification = args.joint_classification
        if args.joint_classification == "sent":
            if args.joint_normalize_before:
                self.joint_layer_norm = LayerNorm(classifier_embed_dim)
            else:
                self.joint_layer_norm = None

            self.joint_layers = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(
                        embedding_dim=classifier_embed_dim,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                    )
                    for _ in range(args.num_joint_layers)
                ]
            )

        self.classifier = RobertaClassificationHead(
            classifier_embed_dim,
            classifier_embed_dim,
            1,  # num_classes
            "tanh",
            args.classifier_dropout,
        )

    def forward(self, src_tokens, src_lengths):
        segment_labels = self.get_segment_labels(src_tokens)
        positions = self.get_positions(src_tokens, segment_labels)

        inner_states, _ = self.model(
            tokens=src_tokens,
            segment_labels=segment_labels,
            last_state_only=True,
            positions=positions,
        )

        return inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C

    def sentence_forward(self, encoder_out, src_tokens=None, sentence_rep="head"):
        # encoder_out: B x T x C
        if sentence_rep == "head":
            x = encoder_out[:, :1, :]
        else:  # 'meanpool', 'maxpool'
            assert src_tokens is not None, "meanpool requires src_tokens input"
            segment_labels = self.get_segment_labels(src_tokens)
            padding_mask = src_tokens.ne(self.padding_idx)
            encoder_mask = segment_labels * padding_mask.type_as(segment_labels)

            if sentence_rep == "meanpool":
                ntokens = torch.sum(encoder_mask, dim=1, keepdim=True)
                x = torch.sum(
                    encoder_out * encoder_mask.unsqueeze(2), dim=1, keepdim=True
                ) / ntokens.unsqueeze(2).type_as(encoder_out)
            else:  # 'maxpool'
                encoder_out[
                    (encoder_mask == 0).unsqueeze(2).repeat(1, 1, encoder_out.shape[-1])
                ] = -float("inf")
                x, _ = torch.max(encoder_out, dim=1, keepdim=True)

        if hasattr(self, "transform_layer"):
            x = self.transform_layer(x)

        return x  # B x 1 x C

    def joint_forward(self, x):
        # x: T x B x C
        if self.joint_layer_norm:
            x = self.joint_layer_norm(x.transpose(0, 1))
            x = x.transpose(0, 1)

        for layer in self.joint_layers:
            x, _ = layer(x, self_attn_padding_mask=None)
        return x

    def classification_forward(self, x):
        # x: B x T x C
        return self.classifier(x)


@dataclass
class DiscriminativeNMTRerankerConfig(FairseqDataclass):
    pretrained_model: str = field(
        default="", metadata={"help": "pretrained model to load"}
    )
    sentence_rep: SENTENCE_REP_CHOICES = field(
        default="head",
        metadata={
            "help": "method to transform the output of the transformer stack to a sentence-level representation"
        },
    )

    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    classifier_dropout: float = field(
        default=0.0, metadata={"help": "classifier dropout probability"}
    )
    embed_dim: int = field(default=768, metadata={"help": "embedding dimension"})
    ffn_embed_dim: int = field(
        default=2048, metadata={"help": "embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    attention_heads: int = field(default=8, metadata={"help": "num attention heads"})
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    apply_bert_init: bool = field(
        default=False, metadata={"help": "use custom param initialization for BERT"}
    )
    activation_fn: ACTIVATION_FN_CHOICES = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    freeze_embeddings: bool = field(
        default=False, metadata={"help": "freeze embeddings in the pretrained model"}
    )
    n_trans_layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": "number of layers to freeze in the pretrained transformer model"
        },
    )

    # joint classfication
    joint_classification: JOINT_CLASSIFICATION_CHOICES = field(
        default="none",
        metadata={"help": "method to compute joint features for classification"},
    )
    num_joint_layers: int = field(
        default=1, metadata={"help": "number of joint layers"}
    )
    joint_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layer norm on the input to the joint layer"},
    )


@register_model(
    "discriminative_nmt_reranker", dataclass=DiscriminativeNMTRerankerConfig
)
class DiscriminativeNMTReranker(BaseFairseqModel):
    @classmethod
    def build_model(cls, args, task):
        model = BertRanker(args, task)
        return DiscriminativeNMTReranker(args, model)

    def __init__(self, args, model):
        super().__init__()

        self.model = model
        self.sentence_rep = args.sentence_rep
        self.joint_classification = args.joint_classification

    def forward(self, src_tokens, src_lengths, **kwargs):
        return self.model(src_tokens, src_lengths)

    def sentence_forward(self, encoder_out, src_tokens):
        return self.model.sentence_forward(encoder_out, src_tokens, self.sentence_rep)

    def joint_forward(self, x):
        return self.model.joint_forward(x)

    def classification_forward(self, x):
        return self.model.classification_forward(x)
