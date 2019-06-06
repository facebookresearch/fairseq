#from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_sentence_encoder_taskemb import init_bert_params
from fairseq.models.transformer_sentence_encoder_taskemb import TransformerSentenceEncoderTaskemb


# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

def set_grads_flag(model, enable):
    for param in model.parameters():
        param.requires_grad = enable


class LMClassifier(nn.Module):

    def __init__(self, args, task):
        super(Classifier, self).__init__()

        dictionary = task.vocab
        self.vocab_size = dictionary.__len__()
        self.padding_idx = dictionary.pad()
        self.encoder_embed_dim = args.encoder_embed_dim
        self.training_mode = args.training_mode
        self.task_emb_size = args.task_emb_size

        self.num_grad_updates = args.num_grad_updates
        self.meta_gradient = args.meta_gradient
        self.normalize_loss = args.normalize_loss
        self.use_momentum = args.use_momentum
        self.log_losses = args.log_losses
        self.z_lr = args.z_lr
        self.reinit_meta_opt = args.reinit_meta_opt
        self.task_emb_init = args.task_emb_init
        self.num_task_examples = args.num_task_examples

        self.sentence_encoder = TransformerSentenceEncoderTaskemb(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=args.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            add_bias_kv=args.bias_kv,
            add_zero_attn=args.zero_attn,
            task_emb_size=args.task_emb_size,
            task_emb_cond_type=args.task_emb_cond_type
        )

        self.classifier = nn.Linear(
            args.encoder_embed_dim, self.vocab_size)

        if not args.tune_model_params:
            print("Model params are not tuned!")
            set_grads_flag(self.sentence_encoder, False)
            set_grads_flag(self.classifier, False)

    def forward(self, src_tokens, task_embedding=None, attn_mask=None, segment_labels=None):

       segment_labels = torch.zeros_like(src_tokens)

       output, _ = self.sentence_encoder(
           src_tokens,
           segment_labels=segment_labels,
           task_embedding=task_embedding,
           self_attn_mask=attn_mask)

       output = output[-1].transpose(0, 1)
       rep_size = output.shape[-1]
       output = output.contiguous().view(-1, rep_size)

       logits = self.classifier(output)

       return logits
