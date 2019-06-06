#from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import higher

from fairseq import utils
from fairseq.models.classifier import Classifier

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_sentence_encoder_taskemb import init_bert_params
from fairseq.models.transformer_sentence_encoder_taskemb import TransformerSentenceEncoderTaskemb


# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.


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


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@register_model('classifier_v2')
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
        self.train_unseen_task = task.train_unseen_task

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

        self.model = Classifier(args, task)

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

        if num_tasks:
            num_ex_per_task = bs // num_tasks

        if split_data:
            if self.train_unseen_task:
                split_ratio = 0.5
            else:
                split_ratio = 0.7

            if num_tasks:
                N_train = int(split_ratio * num_ex_per_task)
                train_mask = torch.cat((torch.ones(num_tasks, N_train), torch.zeros(num_tasks, num_ex_per_task - N_train)), dim=1).cuda()
                train_mask = train_mask.view(-1)
            else:
                N_train = int(split_ratio * bs)
                train_mask = torch.cat((torch.ones(N_train), torch.zeros(bs - N_train)), dim=0).cuda()
            test_mask = 1 - train_mask
        else:
            train_mask = torch.ones(bs).cuda()
            test_mask = train_mask

        segment_labels = torch.zeros_like(src_tokens)

        task_id = src_tokens[:, 0]

        # Strip off task id
        src_tokens = src_tokens[:, 1:]

        outputs = {}

        if ('meta' in self.training_mode or self.training_mode == 'multitask') and (self.training_mode != 'meta_bprop'):
            if mode == 'eval':
                task_embeddings = self.task_embeddings_eval
            else:
                task_embeddings = self.task_embeddings

        # Randomly initialized task embedding
        if self.training_mode == 'multitask':
            task_embedding = task_embeddings(task_id)
        elif self.training_mode == 'single_task':
            task_embedding = self.task_embedding_init
        else:
            task_embedding = None

        if 'meta' in self.training_mode:

            if num_tasks:

                if self.training_mode == 'meta_avginit':
                    if mode == 'eval':
                        mean_task_embedding = self.task_embeddings.weight.data.mean(0)
                        self.task_embeddings_eval.weight.data.copy_(mean_task_embedding)
                    else:
                        task_embedding_sum = self.task_embeddings.weight.data.sum(0)
                        sample_task_embeddings = self.task_embeddings(task_id).data
                        loo_task_embedding_sum = task_embedding_sum.view(1, -1) - sample_task_embeddings
                        sample_mean_task_embeddings = loo_task_embedding_sum / (self.num_train_tasks - 1)
                        self.task_embeddings.weight.data[task_id] = sample_mean_task_embeddings
                        # self.task_embeddings.weight.data[task_id] = self.task_embeddings.weight.data.mean(0)

                elif self.training_mode == 'meta_randinit':
                    if mode == 'eval':
                        self.task_embeddings_eval.weight.data.uniform_()
                    else:
                        self.task_embeddings.weight.data.uniform_()

                elif self.training_mode == 'meta_zeroinit':
                    if mode == 'eval':
                        self.task_embeddings_eval.weight.data.zero_()
                    else:
                        self.task_embeddings.weight.data.zero_()

                elif self.training_mode == 'meta_onesinit':
                    if mode == 'eval':
                        self.task_embeddings_eval.weight.data.fill_(1)
                    else:
                        self.task_embeddings.weight.data.fill_(1)

                elif self.training_mode == 'meta_bprop':
                    task_embeddings = torch.zeros(
                        (num_tasks, self.task_emb_size), requires_grad=True, device=src_tokens.device)

                else:
                    raise ValueError('Unsupported mode')

            if self.training_mode == 'meta_bprop':
                z_optimizer = higher.get_diff_optim(
                   optim.Adam([task_embeddings], lr=self.z_lr),
                   [task_embeddings], device=src_tokens.device
                )
            else:
                if mode == 'eval':
                    z_optimizer = self.z_optimizer_eval
                else:
                    z_optimizer = self.z_optimizer

            step_size = self.z_lr
            set_learning_rate(z_optimizer, step_size)

            losses = []
            for i in range(self.num_grad_updates):

                num_grad_updates = i

                if self.training_mode == 'meta_bprop':
                    task_embedding = task_embeddings.repeat(1, num_ex_per_task).view(bs, -1)
                else:
                    z_optimizer.zero_grad()
                    task_embedding = task_embeddings(task_id)

                logits = self.model(src_tokens, task_embedding)
                loss = compute_loss(logits, targets, normalize_loss=True, mask=train_mask)
                losses.append(loss.item())

                if self.training_mode == 'meta_bprop':
                    task_embeddings, = z_optimizer.step(loss, params=[task_embeddings])
                else:
                    loss.backward()
                    z_optimizer.step()

                if self.log_losses:
                    losses.append(loss.item())

                if i == 0:
                    outputs['pre_accuracy_train'] = compute_accuracy(logits, targets, mask=train_mask)
                    outputs['pre_loss_train'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=train_mask)
                    if split_data:
                        outputs['pre_accuracy_test'] = compute_accuracy(logits, targets, mask=test_mask)
                        outputs['pre_loss_test'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=test_mask)

                    prev_loss = loss.item()
                else:
                    cur_loss = loss.item()

                    if cur_loss > prev_loss:
                        step_size /= 2
                        set_learning_rate(z_optimizer, step_size)
                        if step_size < 1e-6:
                            break

                    prev_loss = cur_loss

            outputs['num_grad_updates'] = 1.0 * num_grad_updates
            if self.log_losses:
                if np.random.uniform() > 0.99:
                    with open(self.log_losses, 'a') as f:
                        losses_str = '%s\n' % ' '.join(map(str, losses))
                        f.write(losses_str)

        # task_embeddings are held fixed in meta modes
        if self.training_mode == 'meta_bprop':
            task_embedding = task_embeddings.repeat(1, num_ex_per_task).view(bs, -1)

        if self.training_mode == 'task_examples':
            src_all_seq_len = src_all_tokens.shape[1]
            src_seq_len = self.max_seq_len + 1
            src_all_num = src_all_seq_len / src_seq_len
            if src_all_num > self.num_task_examples:
                rand_ex_pos = np.random.randint(0, src_all_num - self.num_task_examples)
                src_all_tokens = src_all_tokens[:, rand_ex_pos * src_seq_len: (rand_ex_pos + 1) * src_seq_len]
            all_tokens = torch.cat((src_tokens, src_all_tokens), dim=1)
            segment_labels = torch.zeros_like(all_tokens)
            logits = self.model(all_tokens)
        else:
            logits = self.model(src_tokens, task_embedding.data if 'meta' in self.training_mode else task_embedding)

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
        if "task_embeddings.weight" in state_dict:
            assert state_dict["task_embeddings.weight"].shape[0] == self.num_train_tasks
            mean_task_emb = state_dict["task_embeddings.weight"].mean(0)

        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                print('Ignoring: ', k)
                del state_dict[k]

        if self.task_emb_init == 'mean':
            print('Note: Initializing task embedding with mean')
            state_dict['task_embedding_init'] = mean_task_emb
        elif self.task_emb_init == 'random':
            print('Note: Initializing task embedding randomly')
            state_dict['task_embedding_init'] = torch.randn(self.task_emb_size)
        elif self.task_emb_init == 'zeros':
            print('Note: Initializing task embedding with zeros')
            state_dict['task_embedding_init'] = torch.zeros(self.task_emb_size)
        elif self.task_emb_init == 'uniform':
            print('Note: Initializing task embedding with random uniform')
            state_dict['task_embedding_init'] = torch.FloatTensor(self.task_emb_size).uniform_()

        return state_dict


@register_model_architecture('classifier_v2', 'cls_v2')
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
    args.reinit_meta_opt = getattr(args, 'reinit_meta_opt', False)
