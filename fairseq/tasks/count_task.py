from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data.multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from fairseq.tasks import FairseqTask, register_task


class Dictionary_toy(Dictionary):
    def __init__(self, no_special_tokens=False):
        super().__init__()
        if no_special_tokens:
            self.symbols = []
            self.count = []
            self.indices = {}
            self.nspecial = 0

    @classmethod
    def load_list(cls, load_list):
        d = cls()
        for word in load_list:
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(1)
        return d

    def encode(self, line):
        line_encode = [self.index(word) for word in line]
        return line_encode


@register_task('counting_meta')
class MetaClassificationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--num_train', default=10000, type=int,
                            help='Num training examples')
        parser.add_argument('--num_test', default=10000, type=int,
                            help='Num test examples')
        parser.add_argument('--vocab_size', default=10, type=int,
                            help='Vocabulary size')
        parser.add_argument('--max_tasks', default=5, type=int,
                            help='Number of tasks')
        parser.add_argument('--max_seq_len', default=16, type=int,
                            help='Maximum sequence length')
        parser.add_argument('--toy_task', default='count', type=str,
                            help='Task')
        parser.add_argument('--train_unseen_task', action='store_true',
                            help='Train on unseen task')
        parser.add_argument('--sample_num_tasks', default=1, type=int,
                            help='Num of tasks to sample for each iteration')
        parser.add_argument('--batch_version', action='store_true',
                            help='Batch update')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.

        return MetaClassificationTask(args)

    def __init__(self, args):
        super().__init__(args)
        # self.task = simple_count_task
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.vocab_size = args.vocab_size
        self.max_tasks = args.max_tasks
        self.max_seq_len = args.max_seq_len
        self.toy_task = args.toy_task
        self.train_unseen_task = args.train_unseen_task
        self.sample_num_tasks = args.sample_num_tasks
        self.batch_version = args.batch_version

    def construct_data(self, rng, task_id, num_examples):

        # Read input sentences.
        sentences, lengths, labels = [], [], []

        for j in range(num_examples):

            if self.toy_task == 'count':
                num_occurrences = rng.randint(0, self.max_seq_len)
                num_sample = self.max_seq_len - num_occurrences
                sentence = list(rng.randint(0, self.vocab_size, (num_sample, )))
                sentence += [task_id] * num_occurrences
                rng.shuffle(sentence)
                assert len(sentence) == self.max_seq_len

                label = sentence.count(task_id)

            elif self.toy_task == 'present':
                sentence = rng.randint(0, self.vocab_size, (self.max_seq_len, ))
                if rng.uniform() > 0.5:
                    sentence[0] = task_id
                rng.shuffle(sentence)
                label = int(task_id in sentence)

            sentence = [self.cls_index] + self.input_vocab.encode(sentence)
            sentence = [task_id] + sentence
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        vocab_size = self.vocab_size
        max_tasks = self.max_tasks
        max_seq_len = self.max_seq_len

        self.output_vocab_size = max_seq_len + 1
        # self.output_vocab_size = 2

        input_vocab = Dictionary_toy.load_list(range(vocab_size))
        self.cls_index = input_vocab.add_symbol('cls')
        output_vocab = Dictionary_toy(no_special_tokens=True).load_list(range(max_seq_len))
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        random_seeds = {'train': 1234, 'valid': 2345, 'test': 3456}
        rng = np.random.RandomState(seed=random_seeds[split])

        dataset_map = OrderedDict()

        if split == 'train' and not self.train_unseen_task:
            for i in range(max_tasks - 1):
                task_id = i
                sentences, labels, lengths = self.construct_data(rng, task_id, self.num_train)
                dataset_map[i] = LanguagePairDataset(
                    src=sentences,
                    src_sizes=lengths,
                    src_dict=input_vocab,
                    tgt=labels,
                    tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                    tgt_dict=output_vocab,
                    left_pad_source=False,
                    max_target_positions=1,
                    input_feeding=False
                )
            self.datasets[split] = MultiCorpusSampledDataset(dataset_map,
                num_samples=self.sample_num_tasks)
        else:
            task_id = max_tasks - 1

            sentences, labels, lengths = self.construct_data(rng, task_id, self.num_test)
            self.datasets[split] = LanguagePairDataset(
                src=sentences,
                src_sizes=lengths,
                src_dict=input_vocab,
                tgt=labels,
                tgt_sizes=torch.ones(len(labels)),  # targets have length 1
                tgt_dict=output_vocab,
                left_pad_source=False,
                max_target_positions=1,
                input_feeding=False,
            )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab

    def _split(self, batch, size, k):
        return batch[k * size: (k + 1) * size]

    def _get_loss_tasks(self, sample, model, criterion):

        sample_sub = {}
        bs = sample['target'].shape[0]
        logs = []
        losses = []
        k = self.sample_num_tasks
        for i in range(k):
            sample_sub['target'] = self._split(sample['target'], bs // k, i)
            sample_sub['net_input'] = {}
            sample_sub['net_input']['src_tokens'] = self._split(sample['net_input']['src_tokens'], bs // k, i)
            sample_sub['net_input']['src_lengths'] = self._split(sample['net_input']['src_lengths'], bs // k, i)
            sample_sub['ntokens'] = sample['ntokens']

            loss, sample_size, logging_output = self._get_loss(sample_sub, model, criterion)
            losses.append(loss)
            logs.append(logging_output)

        loss = sum(losses) / k
        log_overall = {}
        for key in logging_output.keys():
            log_overall[key] = sum([log[key] for log in logs]) / k

        return loss, sample_size, log_overall

    def _get_loss(self, sample, model, criterion, eval_mode=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['eval_mode'] = eval_mode

        with torch.set_grad_enabled(True):
            outputs = model(**sample['net_input'])

        loss = outputs['loss']

        #sample_size = sample['target'].size(0)
        sample_size = 1

        logging_output = {
            'loss': loss.item(),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        for diag in ['accuracy', 'pre_accuracy', 'grad_norm', 'loss_delta']:
            if diag in outputs:
                logging_output[diag] = outputs[diag].item()
        if 'num_grad_updates' in logging_output:
            logging_output['num_grad_updates'] = outputs['num_grad_updates']

        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        if self.batch_version:
            sample['net_input']['num_tasks'] = self.sample_num_tasks
            loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        else:
            loss, sample_size, logging_output = self._get_loss_tasks(sample, model, criterion)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        # with torch.no_grad():
        # loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        # Eval mode: Use 25% of the data to validation. The 75% is used for training by meta-learned 
        # models and ignored by non-meta learning models. 
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion, eval_mode=True)

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        for other_metrics in ['accuracy', 'pre_accuracy', 'grad_norm', 'loss_delta', 'num_grad_updates']:
            agg_logging_outputs[other_metrics] = sum(
                log[other_metrics] for log in logging_outputs if other_metrics in log
            )
        return agg_logging_outputs
