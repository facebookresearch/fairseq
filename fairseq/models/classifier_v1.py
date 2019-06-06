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


def compute_accuracy(logits, targets, mask=None):
    predictions = logits.argmax(dim=1)
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    assert predictions.shape == targets.shape
    accuracy = predictions.eq(targets).float()
    if mask is not None:
        accuracy *= mask
        accuracy = accuracy.sum() / mask.sum()
    else:
        accuracy = accuracy.mean()

    return accuracy


def compute_loss(logits, target, normalize_loss=False, mask=None):
    logits = logits.view(-1, logits.size(-1))
    target = target.view(-1)

    if mask is not None:
        loss = F.cross_entropy(logits, target, reduction='none') * mask
        if normalize_loss:
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.sum()
    else:
        loss = F.cross_entropy(logits, target, reduction='mean' if normalize_loss else 'sum')
    return loss


class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, encoder_embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_embed_dim = encoder_embed_dim
        self.embeddings = nn.Embedding(self.vocab_size, self.encoder_embed_dim)
        self.rnn = nn.GRU(self.encoder_embed_dim, self.encoder_embed_dim, 1)

    def forward(self, x, segment_labels, task_embedding):
        word_embeddings = self.embeddings(x)
        task_embedding = task_embedding.view(1, 1, -1)
        bs = x.shape[0]
        task_embeddings = task_embedding.repeat(bs, 1, 1)
        word_embeddings = torch.cat((task_embeddings, word_embeddings), 1)
        hids, sentence_rep = self.rnn(word_embeddings.transpose(0, 1))
        sentence_rep = sentence_rep.squeeze()
        return hids, sentence_rep


@register_model('classifier_v1')
class FairseqTransformerClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--bias-kv', action='store_true',
                            help='if set, adding a learnable bias kv')
        parser.add_argument('--zero-attn', action='store_true',
                            help='if set, pads attn with zero')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                            ' and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--num-segment', type=int, metavar='N',
                            help='num segment in the input')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for pooler layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')

        parser.add_argument('--tune_model_params', action='store_true',
                            help='Tune model params')
        parser.add_argument('--encoder_type', default='transformer', type=str,
                            help='type of encoder: transformer/RNN')
        parser.add_argument('--task_emb_cond_type', default='token', type=str,
                            help='type of encoder: transformer/RNN')
        parser.add_argument('--training_mode', default='multitask', type=str,
                            help='Multi-tasking/meta-learning')
        parser.add_argument('--num_grad_updates', default=1, type=int,
                            help='Number of grad steps in inner loop')
        parser.add_argument('--meta_gradient', action='store_true',
                            help='Backprop through optimization or not')
        parser.add_argument('--regularization', action='store_true',
                            help='Enable/Disable all regularization')
        parser.add_argument('--normalize_loss', action='store_true',
                            help='Normalize loss by # examples')
        parser.add_argument('--use_momentum', action='store_true',
                            help='Use momentum for task embedding updates')
        parser.add_argument('--task_emb_size', default=128, type=int,
                            help='Size of task embedding')

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Return the wrapped version of the module
        return FairseqTransformerClassifier(args, task)

    def __init__(self, args, task):
        super(FairseqTransformerClassifier, self).__init__()

        dictionary = task.input_vocab
        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_tasks = task.max_tasks
        self.encoder_type = args.encoder_type
        self.encoder_embed_dim = args.encoder_embed_dim
        self.training_mode = args.training_mode
        self.num_grad_updates = args.num_grad_updates
        self.meta_gradient = args.meta_gradient
        self.normalize_loss = args.normalize_loss
        self.use_momentum = args.use_momentum
        self.task_emb_size = args.task_emb_size

        if self.training_mode == 'multitask':
            self.task_embeddings = nn.Embedding(
                self.max_tasks, self.task_emb_size, padding_idx=None)

        elif self.training_mode == 'meta_maml' or self.training_mode == 'single_task':
            self.task_embedding_init = nn.Parameter(torch.randn(self.task_emb_size))

        elif self.training_mode == 'meta_previnit' or self.training_mode == 'meta_avginit':
            self.task_embeddings = torch.randn((self.max_tasks, self.task_emb_size)).cuda()

        # elif self.training_mode == 'meta_avginit':
        #     self.task_embedding_inits = torch.randn(
        #         self.max_tasks, self.encoder_embed_dim, requires_grad=True)

        if self.use_momentum:
            self.task_momentum_grads = torch.zeros(
                self.max_tasks, self.task_emb_size).cuda()

        if self.encoder_type == 'transformer':
            self.sentence_encoder = TransformerSentenceEncoderTaskemb(
                padding_idx=self.padding_idx,
                vocab_size=self.vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                task_emb_size=args.task_emb_size,
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
                task_emb_cond_type=args.task_emb_cond_type
            )
        else:
            self.sentence_encoder = RNNEncoder(self.vocab_size, args.encoder_embed_dim)

        self.classifier = nn.Linear(
            args.encoder_embed_dim, task.output_vocab_size)

        if not args.tune_model_params:
            print("Model params are not tuned!")
            set_grads_flag(self.sentence_encoder, False)
            set_grads_flag(self.classifier, False)

    def compute_update(self, logits, targets, parameters, step_size, momentum_grads=None, num_tasks=None, mask=None):

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        if mask is not None:
            loss = F.cross_entropy(logits, targets, reduction='none') * mask
            loss = loss.sum()
            num_samples = mask.sum()
        else:
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_samples = logits.size(0)

        sample_normalized_loss = loss / num_samples

        normalizing_factor = num_samples
        if num_tasks:
            normalizing_factor /= num_tasks
        loss = loss / normalizing_factor

        grads = torch.autograd.grad(loss, parameters, create_graph=self.meta_gradient, retain_graph=False)
        assert len(grads) == 1
        grads = grads[0]
        grad_norm = grads.norm()

        if momentum_grads is not None:
            momentum_grads = 0.7 * momentum_grads + grads
            updated_params = parameters - step_size * momentum_grads
        else:
            updated_params = parameters - step_size * grads

        return updated_params, grad_norm, sample_normalized_loss, momentum_grads

    def forward(self, src_tokens, src_lengths, targets, num_tasks=None, split_data=False):

        bs = src_tokens.shape[0]

        if num_tasks:
            num_ex_per_task = bs // num_tasks

        if split_data:
            if num_tasks:
                N_train = int(0.7 * num_ex_per_task)
                train_mask = torch.cat((torch.ones(N_train, num_tasks), torch.zeros(num_ex_per_task - N_train, num_tasks)), dim=0).cuda()
                train_mask = train_mask.view(-1)
            else:
                N_train = int(0.7 * bs)
                train_mask = torch.cat((torch.ones(N_train), torch.zeros(bs - N_train)), dim=0).cuda()
            test_mask = 1 - train_mask
        else:
            train_mask = torch.ones(bs).cuda()
            test_mask = train_mask

        segment_labels = torch.zeros_like(src_tokens)

        task_id = src_tokens[:, 0]

        # Strip off task id
        src_tokens = src_tokens[:, 1:]

        if self.encoder_type == 'rnn':
            src_tokens = torch.flip(src_tokens, (1,))

        outputs = {}

        # Randomly initialized task embedding
        if self.training_mode == 'multitask':
            task_embedding = self.task_embeddings(task_id)
        elif self.training_mode == 'meta_randinit':
            task_embedding = torch.randn(self.task_emb_size, requires_grad=True).cuda()
        elif self.training_mode == 'meta_zeroinit':
            task_embedding = torch.zeros(self.task_emb_size, requires_grad=True).cuda()
        elif self.training_mode == 'meta_maml' or self.training_mode == 'single_task':
            task_embedding = self.task_embedding_init

        if 'meta' in self.training_mode:
            grad_norms = []

            if num_tasks:
                unique_task_ids = task_id.view(-1, num_tasks)[0]
                if self.training_mode == 'meta_previnit':
                    task_embedding = self.task_embeddings[unique_task_ids]
                    task_embedding.requires_grad = True
                elif self.training_mode == 'meta_avginit':
                    task_embedding = self.task_embeddings[unique_task_ids].mean(0)
                    task_embedding = task_embedding.view(1, -1).expand(num_tasks, -1)
                    task_embedding.requires_grad = True
                else:
                    task_embedding = task_embedding.view(1, -1).expand(num_tasks, -1)

            num_grad_updates = self.num_grad_updates

            if self.use_momentum:
                momentum_grads = self.task_momentum_grads[unique_task_ids]
            else:
                momentum_grads = None

            step_size = 1.0

            for i in range(self.num_grad_updates):

                if num_tasks:
                    task_embedding_per_example = task_embedding.repeat(1, num_ex_per_task).view(bs, -1)
                else:
                    task_embedding_per_example = task_embedding

                _, sentence_rep = self.sentence_encoder(
                    src_tokens, segment_labels, task_embedding_per_example)
                logits = self.classifier(sentence_rep)
                task_embedding_updated, grad_norm, loss, momentum_grads = self.compute_update(
                    logits, targets, task_embedding, step_size, momentum_grads=momentum_grads, num_tasks=num_tasks, mask=train_mask)

                if i == 0:
                    outputs['pre_accuracy_train'] = compute_accuracy(logits, targets, mask=train_mask)
                    outputs['pre_loss_train'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=train_mask)
                    if split_data:
                        outputs['pre_accuracy_test'] = compute_accuracy(logits, targets, mask=test_mask)
                        outputs['pre_loss_test'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=test_mask)

                    prev_best_loss = loss.item()
                    prev_good_task_embedding = task_embedding
                    task_embedding = task_embedding_updated

                else:
                    cur_loss = loss.item()

                    if cur_loss > prev_best_loss:
                        task_embedding = prev_good_task_embedding
                        step_size /= 2
                        if step_size < 1e-3:
                            num_grad_updates = i
                            break
                    else:
                        prev_good_task_embedding = task_embedding
                        task_embedding = task_embedding_updated

                grad_norms.append(grad_norm)

            if self.use_momentum:
                self.task_momentum_grads[unique_task_ids] = momentum_grads

            if self.training_mode == 'meta_previnit':
                self.task_embeddings[unique_task_ids] = task_embedding.data

            grad_norm = sum(grad_norms) / self.num_grad_updates

            outputs['grad_norm'] = grad_norm
            outputs['num_grad_updates'] = 1.0 * num_grad_updates

        if ('meta' in self.training_mode) and num_tasks:
            task_embedding_per_example = task_embedding.repeat(1, num_ex_per_task).view(bs, -1)
        else:
            task_embedding_per_example = task_embedding

        _, sentence_rep = self.sentence_encoder(
            src_tokens, segment_labels, task_embedding_per_example)
        logits = self.classifier(sentence_rep)

        outputs['post_accuracy_train'] = compute_accuracy(logits, targets, mask=train_mask)
        outputs['post_loss_train'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=train_mask)
        if 'pre_loss_train' in outputs:
            outputs['train_loss_delta'] = outputs['pre_loss_train'] - outputs['post_loss_train']
        if split_data:
            outputs['post_accuracy_test'] = compute_accuracy(logits, targets, mask=test_mask)
            outputs['post_loss_test'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=test_mask)
            if 'pre_loss_test' in outputs:
                outputs['test_loss_delta'] = outputs['pre_loss_test'] - outputs['post_loss_test']

        return outputs

    def upgrade_state_dict_named(self, state_dict, name):
        #state_dict['task_embedding_init'] = state_dict["task_embeddings.weight"].mean(0)
        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                del state_dict[k]
        state_dict['task_embedding_init'] = torch.randn(self.task_emb_size)
        #state_dict['task_embeddings.weight'] = torch.randn(self.max_tasks, self.encoder_embed_dim)
        return state_dict


@register_model_architecture('classifier_v1', 'cls_v1')
def toy_transformer_cls(args):

    args.regularization = getattr(args, 'regularization', False)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)

    if not args.regularization:
        args.dropout = 0.0
        args.attention_dropout = 0.0
        args.act_dropout = 0.0

    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.bias_kv = getattr(args, 'bias_kv', False)
    args.zero_attn = getattr(args, 'zero_attn', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.task_emb_size = getattr(args, 'task_emb_size', 128)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.num_segment = getattr(args, 'num_segment', 2)

    args.apply_bert_init = getattr(args, 'apply_bert_init', False)

    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)

    args.tune_model_params = getattr(args, 'tune_model_params', False)
    args.meta_gradient = getattr(args, 'meta_gradient', False)
    args.use_momentum = getattr(args, 'use_momentum', False)
