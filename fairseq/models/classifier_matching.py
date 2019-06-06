from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fairseq import utils
from fairseq.models.classifier import Classifier

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_sentence_encoder_taskemb import init_bert_params
from fairseq.models.transformer_sentence_encoder_taskemb import TransformerSentenceEncoderTaskemb


def compute_accuracy(logits, src_labels, query_labels, mask=None):

    assert src_labels.shape[0] == query_labels.shape[0]

    logits = logits.view(query_labels.shape[0], -1)

    num_classes = 4

    logprobs = []

    for i in range(num_classes):

        target_mask = src_labels.eq(i)
        target_logits = logits.masked_fill((1 - target_mask).byte(), float('-inf'))
        logprobs.append(target_logits.logsumexp(-1, keepdim=True))

    logprobs = torch.cat(logprobs, -1)

    predictions = logprobs.argmax(-1)

    predictions = predictions.view(-1)
    query_labels = query_labels.view(-1)

    accuracy = (predictions == query_labels).float()

    if mask is not None:
        mask = mask.view(-1)
        accuracy = (accuracy * mask).sum() / mask.sum()
    else:
        accuracy = accuracy.mean()

    return accuracy


def compute_loss(logits, src_labels, query_labels, mask=None):

    assert src_labels.shape[0] == query_labels.shape[0]

    logits = logits.view(query_labels.shape[0], -1)

    target_mask = src_labels.eq(query_labels)

    target_logits = logits.masked_fill((1 - target_mask).byte(), float('-inf'))

    loss = -target_logits.logsumexp(-1) + logits.logsumexp(-1)

    inf_mask = (target_mask.sum(-1) > 0).float()

    loss.masked_fill_((1 - inf_mask).byte(), 0)

    if mask is not None:
        mask = mask.view(-1)
        return (loss * mask).sum() / ((mask * inf_mask).sum() + 1e-9)

    return loss.sum() / (inf_mask.sum() + 1e-9)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@register_model('classifier_matching')
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
        parser.add_argument('--log_losses', default=None, type=str,
                            help='Output z optimization losses')
        parser.add_argument('--z_lr', default=1e-3, type=float,
                            help='learning rate for optimizing z')
        parser.add_argument('--reinit_meta_opt', action='store_true',
                            help='Re-initialize meta opt for every step')
        parser.add_argument('--task_emb_init', default='mean', type=str,
                            help='How to initialize rask embedding.')
        parser.add_argument('--num_task_examples', default=100, type=int,
                            help='Number of examples in task description.')
        parser.add_argument('--meta_num_ex', default=11, type=int,
                            help='Number of examples to use for meta learning.')
        parser.add_argument('--supervision_at_end', action='store_true',
                            help='Provide supervision only at the end.')
        parser.add_argument('--contextualized', action='store_true',
                            help='Contextualize examples.')
        parser.add_argument('--encoder_layers', default=1, type=int,
                            help='Number of encoder layers.')

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
        self.log_losses = args.log_losses
        self.z_lr = args.z_lr
        self.reinit_meta_opt = args.reinit_meta_opt
        self.num_train_tasks = task.num_train_tasks
        self.num_test_tasks = task.num_test_tasks
        self.task_emb_init = args.task_emb_init
        self.num_task_examples = args.num_task_examples
        self.max_seq_len = task.max_seq_len
        self.meta_num_ex = args.meta_num_ex
        self.supervision_at_end = args.supervision_at_end
        self.task = task
        self.contextualized = args.contextualized

        self.model = Classifier(args, task)

        if self.training_mode == 'multitask':
            self.task_embeddings = nn.Embedding(
                self.num_train_tasks, self.task_emb_size)
            self.task_embeddings_eval = nn.Embedding(
                self.num_test_tasks, self.task_emb_size)

        elif 'meta' in self.training_mode and self.training_mode != 'meta_bprop':
            # Train
            self.task_embeddings = nn.Embedding(
                self.num_train_tasks, self.task_emb_size)
            self.z_optimizer = optim.Adam(
                self.task_embeddings.parameters(), lr=self.z_lr)
            # Eval
            self.task_embeddings_eval = nn.Embedding(
                self.num_test_tasks, self.task_emb_size)
            self.z_optimizer_eval = optim.Adam(
                self.task_embeddings_eval.parameters(), lr=self.z_lr)

        elif self.training_mode == 'single_task':
            self.task_embedding_init = nn.Parameter(torch.randn(self.task_emb_size))

    def forward(
        self,
        src_tokens,
        src_lengths,
        targets,
        src_all_tokens=None,
        num_tasks=None,
        split_data=False,
        optimizer=None,
        mode='train'
    ):
        bs = src_tokens.shape[0]

        train_mask, test_mask = None, None

        if self.task.train_unseen_task and mode == 'eval':

            pair_seq_len = 2 * self.task.max_seq_len + 1
            src_tokens = src_tokens.view(-1, pair_seq_len)

            bs = src_tokens.shape[0]

            cls_tensor = torch.LongTensor([self.task.cls_encode]).view(1, -1).expand(bs, 1).cuda()

            src_labels = targets[:, :-1]
            query_labels = targets[:, [-1]]

            split_seq_len = src_labels.shape[1]

            if self.contextualized:
                src_tokens_ctxt = src_tokens[:, self.task.max_seq_len:]
                src_tokens_ctxt = src_tokens_ctxt.contiguous().view(bs // split_seq_len, -1).repeat(1, split_seq_len)
                src_tokens_ctxt = src_tokens_ctxt.view(bs, -1)
                src_tokens = torch.cat((src_tokens[:, :-1], src_tokens_ctxt), -1)
            else:
                src_tokens = src_tokens[:, :-1]

            src_tokens = torch.cat((cls_tensor, src_tokens), -1)

        else:

            task_id = src_tokens[:, 0]
            src_tokens = src_tokens[:, 1:]

            if self.task.train_unseen_task:
                query_tokens = src_tokens

                random_indices = torch.LongTensor(bs ** 2).random_(0, bs)
                src_tokens = src_tokens[random_indices]

                if self.contextualized:
                    src_tokens_ctxt = src_tokens.view(bs, -1).repeat(1, bs).view(bs ** 2, -1)
                    src_tokens = torch.cat((src_tokens[:, :-1], src_tokens_ctxt), -1)
                else:
                    src_tokens = src_tokens[:, :-1]

                query_tokens = query_tokens.repeat(1, bs).view(bs ** 2, -1)
                query_tokens = query_tokens[:, :-1]

                query_labels = targets.view(-1, 1)
                src_labels = targets[random_indices].view(-1, bs)

                bs = bs ** 2

            else:

                num_ex_per_task = bs // num_tasks
                src_tokens = src_tokens.view(num_tasks, bs // num_tasks, -1)
                targets = targets.view(num_tasks, -1)
                task_id = task_id.view(num_tasks, -1)

                split_seq_len = self.meta_num_ex

                if num_ex_per_task < split_seq_len:

                    random_example_order = torch.LongTensor(split_seq_len).random_(0, num_ex_per_task)
                    num_ex_per_task = split_seq_len
                    # print('Batch size too small')
                    # split_seq_len = num_ex_per_task
                else:
                    random_example_order = torch.LongTensor(np.random.permutation(num_ex_per_task))

                src_tokens = src_tokens[:, random_example_order]
                targets = targets[:, random_example_order]
                task_id = task_id[:, random_example_order]

                d = num_ex_per_task // split_seq_len
                num_ex_per_task = d * split_seq_len

                src_tokens = src_tokens[:, :num_ex_per_task].contiguous()
                targets = targets[:, :num_ex_per_task].contiguous()
                task_id = task_id[:, :num_ex_per_task].contiguous()

                random_queries = torch.LongTensor(d).random_(0, num_ex_per_task)

                if split_data:
                    split_ratio = 0.5
                    N_train = int(split_ratio * d)
                    train_mask = torch.cat((torch.ones(num_tasks, N_train), torch.zeros(num_tasks, d - N_train)), dim=1).cuda()
                    train_mask = train_mask.view(-1)
                    test_mask = 1 - train_mask
                    if N_train == 0:
                        train_mask.fill_(1)
                        test_mask.fill_(1)

                query_tokens = src_tokens[:, random_queries]
                query_tokens = query_tokens.repeat(1, 1, split_seq_len)
                query_labels = targets[:, random_queries]

                bs = num_tasks * num_ex_per_task

                src_tokens = src_tokens.view(bs, -1)
                query_tokens = query_tokens.view(bs, -1)

                assert src_tokens.shape[-1] == query_tokens.shape[-1]

                if self.contextualized:
                    query_tokens = query_tokens[:, :-1]

                    src_tokens_context = src_tokens.view(bs // split_seq_len, -1)
                    src_tokens_context = src_tokens_context.repeat(1, split_seq_len)
                    src_tokens_context = src_tokens_context.view(bs, -1)
                    src_tokens = torch.cat((src_tokens[:, :-1], src_tokens_context), -1)
                else:
                    query_tokens = query_tokens[:, :-1]
                    src_tokens = src_tokens[:, :-1]

                src_labels = targets.view(-1, split_seq_len)
                query_labels = query_labels.view(-1, 1)

            task_id = task_id.view(-1)

            cls_tensor = torch.LongTensor([self.task.cls_encode]).view(1, -1).repeat(bs, 1).cuda()
            pair_tokens = torch.cat((cls_tensor, query_tokens, src_tokens), -1)

            src_tokens = pair_tokens

        outputs = {}

        if ('meta' in self.training_mode) or (self.training_mode == 'multitask'):
            if mode == 'eval':
                task_embeddings = self.task_embeddings_eval
            else:
                task_embeddings = self.task_embeddings

        if 'meta' in self.training_mode:
            if mode == 'eval':
                self.task_embeddings_eval.weight.data.zero_()
                z_optimizer = self.z_optimizer_eval
            else:
                self.task_embeddings.weight.data.zero_()
                z_optimizer = self.z_optimizer

            step_size = self.z_lr
            num_grad_updates = 0
            for i in range(self.num_grad_updates):

                num_grad_updates = i

                z_optimizer.zero_grad()
                task_embedding = task_embeddings(task_id)

                logits = self.model(src_tokens, task_embedding=task_embedding)
                loss = compute_loss(logits, src_labels, query_labels, mask=train_mask)

                loss.backward()
                z_optimizer.step()

                if i == 0:
                    outputs['pre_accuracy_train'] = compute_accuracy(logits, src_labels, query_labels, mask=train_mask)
                    outputs['pre_loss_train'] = compute_loss(logits, src_labels, query_labels, mask=train_mask)
                    if split_data:
                        outputs['pre_accuracy_test'] = compute_accuracy(logits, src_labels, query_labels, mask=test_mask)
                        outputs['pre_loss_test'] = compute_loss(logits, src_labels, query_labels, mask=test_mask)

                    prev_loss = loss.item()
                else:
                    cur_loss = loss.item()

                    if cur_loss > prev_loss:
                        step_size /= 2
                        set_learning_rate(z_optimizer, step_size)
                        if step_size < 1e-6:
                            break

                    prev_loss = cur_loss
            outputs['num_grad_updates'] = num_grad_updates

        if self.training_mode == 'multitask':
            task_embedding = task_embeddings(task_id)
        elif 'meta' in self.training_mode:
            task_embedding = task_embeddings(task_id)
        elif self.training_mode == 'single_task':
            task_embedding = self.task_embedding_init
        else:
            assert 'matching' in self.training_mode
            task_embedding = None

        logits = self.model(
            src_tokens,
            task_embedding=task_embedding.data if 'meta' in self.training_mode else task_embedding)

        outputs['post_accuracy_train'] = compute_accuracy(logits, src_labels, query_labels, mask=train_mask)
        outputs['post_loss_train'] = compute_loss(logits, src_labels, query_labels, mask=train_mask)
        if split_data:
            outputs['post_accuracy_test'] = compute_accuracy(logits, src_labels, query_labels, mask=test_mask)
            outputs['post_loss_test'] = compute_loss(logits, src_labels, query_labels, mask=test_mask)

        return outputs

    def upgrade_state_dict_named(self, state_dict, name):

        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                print('Ignoring: ', k)
                del state_dict[k]

        print('Note: Initializing task embedding with zeros')
        if 'matching' not in self.training_mode:
            state_dict['task_embedding_init'] = torch.zeros(self.task_emb_size)

        return state_dict


@register_model_architecture('classifier_matching', 'cls_match')
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
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
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
    args.reinit_meta_opt = getattr(args, 'reinit_meta_opt', False)
    args.supervision_at_end = getattr(args, 'supervision_at_end', False)
    args.contextualized = getattr(args, 'contextualized', False)
