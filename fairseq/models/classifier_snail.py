# from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import higher

from fairseq import utils
from fairseq.models.classifier import Classifier

from fairseq.models import ( BaseFairseqModel,
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


def subsequent_mask(size):
    "Mask out subsequent positions."
    # attn_shape = (size, size)
    # subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # return torch.from_numpy(subsequent_mask) == 0
    # return torch.triu(torch.Tensor(size, size).fill_(float('-inf')), 1)
    # return torch.triu(torch.ones(size, size), 1)
    return torch.tril(torch.ones(size, size))


class SnailClassifier(Classifier):

    def __init__(self, args, task, last_n_outputs):
        super(SnailClassifier, self).__init__(args, task)
        self.last_n_outputs = last_n_outputs

    def forward(
        self,
        src_tokens,
        task_embedding=None,
        attn_mask=None,
        segment_labels=None,
        cls_mask=None
    ):

        segment_labels = torch.zeros_like(src_tokens)

        output, _ = self.sentence_encoder(
            src_tokens,
            segment_labels=segment_labels,
            task_embedding=task_embedding,
            self_attn_mask=attn_mask,
            cls_mask=cls_mask)

        output = output[-1].transpose(0, 1)

        rep_size = output.shape[-1]

        cls_outputs = output[:, -self.last_n_outputs:]
        cls_outputs = cls_outputs.contiguous().view(-1, rep_size)

        logits = self.classifier(cls_outputs)

        return logits


@register_model('classifier_sequence')
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

        if task.train_unseen_task:
            last_n_outputs = 1
        elif self.supervision_at_end:
            last_n_outputs = 1
        else:
            last_n_outputs = self.meta_num_ex

        self.model = SnailClassifier(args, task, last_n_outputs)

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

        src_tokens = src_tokens.view(-1, self.max_seq_len + 2)
        task_id = src_tokens[:, 0]
        src_tokens = src_tokens[:, 1:].contiguous()

        src_tokens = src_tokens.view(bs, -1)

        single_seq_len = src_tokens.shape[1]

        if num_tasks:
            num_ex_per_task = bs // num_tasks

        if split_data and self.task.train_unseen_task:

            split_ratio = 0.5

            if num_tasks:
                N_train = int(split_ratio * num_ex_per_task)
                train_mask = torch.cat((torch.ones(num_tasks, N_train), torch.zeros(num_tasks, num_ex_per_task - N_train)), dim=1).cuda()
                train_mask = train_mask.view(-1)
            else:
                N_train = int(split_ratio * bs)
                train_mask = torch.cat((torch.ones(N_train), torch.zeros(bs - N_train)), dim=0).cuda()
            if bs == 1:
                train_mask, test_mask = 1, 1
            else:
                test_mask = 1 - train_mask
        else:
            train_mask, test_mask = None, None

        outputs = {}

        if not self.task.train_unseen_task:

            num_ex_per_task = bs // num_tasks

            src_tokens = src_tokens.view(num_tasks, bs // num_tasks, -1)
            targets = targets.view(num_tasks, -1)
            task_id = task_id.view(num_tasks, -1)

            random_example_order = torch.LongTensor(np.random.permutation(num_ex_per_task))

            src_tokens = src_tokens[:, random_example_order]
            targets = targets[:, random_example_order]
            task_id = task_id[:, random_example_order]

            split_seq_len = self.meta_num_ex

            if num_ex_per_task < split_seq_len:
                print('Batch size too small')
                split_seq_len = num_ex_per_task

            d = num_ex_per_task // split_seq_len
            num_ex_per_task = d * split_seq_len

            src_tokens = src_tokens[:, :num_ex_per_task].contiguous()
            targets = targets[:, :num_ex_per_task].contiguous()
            task_id = task_id[:, :num_ex_per_task].contiguous()

            bs = num_tasks * num_ex_per_task

            new_bs = bs // split_seq_len
            src_tokens = src_tokens.view(new_bs, -1)
            src_tokens_seq_len = src_tokens.shape[1]

            task_id = task_id.view(new_bs, -1)[:, 0]

            cls_tensor = split_seq_len * [self.task.cls_encode]
            cls_tensor = torch.LongTensor(cls_tensor).view(1, -1).repeat(new_bs, 1).cuda()
            src_tokens = torch.cat((src_tokens, cls_tensor), -1)

            cls_mask = torch.cat((torch.zeros(src_tokens_seq_len), torch.ones(split_seq_len)), 0).cuda()

            mask_positions = (torch.arange(split_seq_len) + 1) * single_seq_len - 1
            mask_positions = torch.nn.functional.one_hot(mask_positions, src_tokens_seq_len)
            query_mask = 1 - torch.cumsum(mask_positions, -1)

            assert query_mask[-1, -2] == 1
            assert query_mask[-1, -1] == 0

            query_mask = torch.cat((query_mask.float(), torch.eye(split_seq_len).float()), -1)

            instance_mask = subsequent_mask(src_tokens_seq_len)

            instance_mask = torch.cat((instance_mask, torch.zeros(src_tokens_seq_len, split_seq_len)), -1)
            attn_mask = torch.cat((instance_mask, query_mask), 0).cuda()

            attn_mask = 1 - attn_mask

            attn_mask.masked_fill_(attn_mask.byte(), float('-inf'))

        else:

            if mode == 'train':

                src_tokens = src_tokens.view(bs, -1, self.max_seq_len + 1)
                query_tokens = src_tokens[:, [-1]]
                src_tokens = src_tokens[:, :-1]
                num_src = src_tokens.shape[1]
                random_example_order = torch.LongTensor(np.random.permutation(num_src))
                src_tokens = src_tokens[:, random_example_order]
                src_tokens = torch.cat((src_tokens, query_tokens), 1).view(bs, -1)

            cls_tensor = torch.LongTensor([self.task.cls_encode]).view(1, -1).expand(bs, 1).cuda()
            # Replace last label with cls token
            src_tokens = torch.cat((src_tokens[:, :-1], cls_tensor), -1)

            attn_mask = subsequent_mask(single_seq_len).float().cuda()
            attn_mask = 1 - attn_mask
            attn_mask.masked_fill_(attn_mask.byte(), float('-inf'))

            cls_mask = torch.cat((torch.zeros(single_seq_len - 1), torch.ones(1)), 0).cuda()

        if self.task.train_unseen_task:
            targets = targets.view(-1)
        elif self.supervision_at_end:
            targets = targets.view(new_bs, -1)[:, -1]
        else:
            targets = targets.view(-1)

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

                logits = self.model(src_tokens, attn_mask=attn_mask, cls_mask=cls_mask, task_embedding=task_embedding)
                loss = compute_loss(logits, targets, normalize_loss=True, mask=train_mask)

                loss.backward()
                z_optimizer.step()

                if i == 0:
                    outputs['pre_accuracy_train'] = compute_accuracy(logits, targets, mask=train_mask)
                    outputs['pre_loss_train'] = compute_loss(logits, targets, normalize_loss=True, mask=train_mask)
                    if split_data:
                        outputs['pre_accuracy_test'] = compute_accuracy(logits, targets, mask=test_mask)
                        outputs['pre_loss_test'] = compute_loss(logits, targets, normalize_loss=True, mask=test_mask)

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
            assert 'snail' in self.training_mode
            task_embedding = None

        logits = self.model(
            src_tokens,
            attn_mask=attn_mask,
            cls_mask=cls_mask,
            task_embedding=task_embedding.data if 'meta' in self.training_mode else task_embedding)

        outputs['post_accuracy_train'] = compute_accuracy(logits, targets, mask=train_mask)
        outputs['post_loss_train'] = compute_loss(logits, targets, normalize_loss=True, mask=train_mask)
        if split_data:
            outputs['post_accuracy_test'] = compute_accuracy(logits, targets, mask=test_mask)
            outputs['post_loss_test'] = compute_loss(logits, targets, normalize_loss=self.normalize_loss, mask=test_mask)

        return outputs

    def upgrade_state_dict_named(self, state_dict, name):

        for k in list(state_dict.keys()):
            print(k)
            if "task_embedding" in k:
                print('Ignoring: ', k)
                del state_dict[k]

        print('Note: Initializing task embedding with zeros')
        if 'snail' not in self.training_mode:
            state_dict['task_embedding_init'] = torch.zeros(self.task_emb_size)

        return state_dict


@register_model_architecture('classifier_sequence', 'cls_seq')
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
