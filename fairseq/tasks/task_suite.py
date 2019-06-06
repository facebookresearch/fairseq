from collections import OrderedDict
import numpy as np
import torch

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


primes_below_1000 = []
for i in range(2, 1000):
    isprime = True
    for prime in primes_below_1000:
        if i % prime == 0:
            isprime = False
    if isprime:
        primes_below_1000.append(i)


def sample_task(seqlen, vocab_size, rng):

    task_description = []

    seq_trans = rng.randint(0, 7)

    if seq_trans == 0:
        # Sorting
        task_description.append('sort')

        def trans_fn(x):
            return sorted(x)
    elif seq_trans == 1:
        # Multiples of k
        k = rng.choice([2, 3, 5, 7, 11])
        task_description.append('multiples of %d' % k)

        def trans_fn(x):
            return [e for e in x if e % k == 0]

    elif seq_trans == 2:
        task_description.append('odd numbers')

        def trans_fn(x):
            return [e for e in x if e % 2 != 0]

    elif seq_trans == 3:
        # Numbers greater than k
        k = rng.randint(0, vocab_size // 2)
        task_description.append('greater than %d' % k)

        def trans_fn(x):
            return [e for e in x if e >= k]

    elif seq_trans == 4:
        # Numbers smaller than k
        k = rng.randint(vocab_size // 2, vocab_size)
        task_description.append('smaller than %d' % k)

        def trans_fn(x):
            return [e for e in x if e <= k]

    elif seq_trans == 5:
        # Primes
        task_description.append('primes')

        def trans_fn(x):
            return [e for e in x if e in primes_below_1000]
    else:
        # Identity
        def trans_fn(x):
            return x

    if seq_trans == 0:  # Sorting
        selection = rng.randint(0, 3)
        if selection == 0:
            task_description.append('first elem')

            def select_fn(x):
                return x[0]
        elif selection == 1:
            task_description.append('last elem')

            def select_fn(x):
                return x[-1]
        else:
            task_description.append('middle elem')

            def select_fn(x):
                return x[len(x) // 2]
    elif seq_trans == 6:  # Identity
        selection = rng.randint(0, 2)
        if selection == 0:
            # Detection
            k = rng.randint(1, seqlen // 2)
            pattern = list(rng.randint(0, vocab_size, k))
            task_description.append('detect:' + ' '.join(map(str, pattern)))

        elif selection == 1:
            # Counting
            pattern = rng.randint(0, vocab_size)
            task_description.append('count: ' + str(pattern))

    else:  # Subsequence
        selection = rng.randint(0, 6)
        if selection == 0:
            # Count
            task_description.append('number of elems')

            def select_fn(x):
                return len(x)
        elif selection == 1:
            # Return first element
            task_description.append('first elem')

            def select_fn(x):
                return x[0]
        elif selection == 2:
            # Return last element
            task_description.append('last elem')

            def select_fn(x):
                return x[-1]
        elif selection == 3:
            # Return middle element
            task_description.append('middle elem')

            def select_fn(x):
                return x[len(x) // 2]
        elif selection == 4:
            # Return min
            task_description.append('min')

            def select_fn(x):
                return min(x)
        else:
            # Return max
            task_description.append('max')

            def select_fn(x):
                return max(x)

    offset = 0
    if seq_trans == 6:
        if type(pattern) == list:
            offset = rng.randint(0, vocab_size - 2)
            task = ['detect', pattern, (0, 1), offset]
        else:
            if vocab_size > seqlen:
                offset = rng.randint(0, vocab_size - seqlen - 1)
            task = ['count', pattern, (0, seqlen), offset]
    else:
        task = [trans_fn, select_fn, (0, vocab_size - 1), 0]

    task_description = '->'.join(task_description)
    return task, task_description


def transform_data(task, data, seqlen, vocab_size, rng):

    task_description = task[1]
    task = task[0]

    data_input = []
    data_output = []

    for instance in data:
        if task[0] == 'detect':
            pattern = task[1]
            start_pos = rng.randint(0, seqlen - len(pattern))
            if rng.randint(0, 2) == 0:
                instance = instance[:start_pos] + pattern + instance[start_pos + len(pattern):]
                assert len(instance) == seqlen
                data_input.append(instance)
                data_output.append(1)
            else:
                data_input.append(instance)
                data_output.append(0)
        elif task[0] == 'count':
            pattern = task[1]
            num_occurrences = rng.randint(0, seqlen)
            instance = [pattern] * num_occurrences + list(rng.randint(0, vocab_size, seqlen - num_occurrences))
            rng.shuffle(instance)
            data_input.append(instance)
            data_output.append(instance.count(pattern))
        else:
            trans_fn, select_fn = task[0], task[1]
            instance_transform = trans_fn(instance)
            while not instance_transform:
                instance = list(rng.randint(0, vocab_size, seqlen))
                instance_transform = trans_fn(instance)
            data_output.append(select_fn(instance_transform))
            data_input.append(instance)

    return data_input, data_output


def precompute_tasks(seqlen, vocab_size, max_tasks):
    rng = np.random.RandomState(12345)
    tasks = []
    task_descriptions = set()
    num_tasks = 0
    while True:
        if num_tasks >= max_tasks:
            break
        task, task_description = sample_task(seqlen, vocab_size, rng)
        if task_description not in task_descriptions:
            tasks.append((task, task_description))
            task_descriptions.add(task_description)
            num_tasks += 1

    rng.shuffle(tasks)

    return tasks


@register_task('task_suite')
class TaskSuite(FairseqTask):

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
                            help='Max tasks to precompute')
        parser.add_argument('--num_train_tasks', default=5, type=int,
                            help='Number of training tasks')
        parser.add_argument('--num_test_tasks', default=5, type=int,
                            help='Number of test tasks')
        parser.add_argument('--max_seq_len', default=16, type=int,
                            help='Maximum sequence length')
        parser.add_argument('--train_unseen_task', action='store_true',
                            help='Train on unseen task')
        parser.add_argument('--sample_num_tasks', default=1, type=int,
                            help='Num of tasks to sample for each iteration')
        parser.add_argument('--batch_version', action='store_true',
                            help='Batch update')
        parser.add_argument('--task_descriptions_dir', default='/tmp', type=str,
                            help='Location to write task descriptions')
        parser.add_argument('--eval_task_id', default=0, type=int,
                            help='Identifier of meta eval task')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.

        return TaskSuite(args)

    def __init__(self, args):
        super().__init__(args)
        # self.task = simple_count_task
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.vocab_size = args.vocab_size
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.max_seq_len = args.max_seq_len
        self.train_unseen_task = args.train_unseen_task
        self.sample_num_tasks = args.sample_num_tasks
        self.batch_version = args.batch_version
        self.eval_task_id = args.eval_task_id

        self.max_tasks = args.max_tasks
        assert self.num_train_tasks + self.num_test_tasks < self.max_tasks
        self.precomputed_tasks = precompute_tasks(
            self.max_seq_len, self.vocab_size, self.max_tasks)

        with open('%s/tasks.txt' % args.task_descriptions_dir, 'w') as f:
            task_descriptions = [task[1] for task in self.precomputed_tasks]
            f.write('\n'.join(task_descriptions))

    def construct_data(self, rng, task_id, task, num_examples):

        data = rng.randint(self.vocab_size, size=(num_examples, self.max_seq_len))
        data = [list(instance) for instance in data]

        data_input, data_output = transform_data(
            task, data, self.max_seq_len, self.vocab_size, rng)

        permitted_output_range = task[0][2]
        permitted_output_offset = task[0][3]
        assert len(permitted_output_range) == 2
        permitted_outputs = list(range(permitted_output_range[0], permitted_output_range[1] + 1))

        # Read input sentences.
        sentences, lengths, labels = [], [], []

        for j in range(num_examples):

            sentence = data_input[j]
            output = data_output[j]

            incorrect_label = rng.choice([0, 1])
            if incorrect_label:
                candidate = rng.choice(permitted_outputs)
                while candidate == output:
                    candidate = rng.choice(permitted_outputs)
                output = candidate
                label = 0
            else:
                label = 1

            output += permitted_output_offset
            assert output < self.vocab_size

            sentence = sentence + [output]
            sentence = [self.cls_index] + self.input_vocab.encode(sentence)
            sentence = [task_id] + sentence
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        print('Constructing data: %s' % split)

        vocab_size = self.vocab_size
        max_seq_len = self.max_seq_len

        self.output_vocab_size = 2

        input_vocab = Dictionary_toy.load_list(range(vocab_size))
        self.cls_index = input_vocab.add_symbol('cls')
        output_vocab = Dictionary_toy(no_special_tokens=True).load_list(range(max_seq_len))
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        data_random_seeds = {'train': 1234, 'valid': 2345, 'test': 3456}

        rng = np.random.RandomState(seed=data_random_seeds[split])

        num_instances_split = {'train': self.num_train, 'valid': self.num_test, 'test': self.num_test}
        num_instances = num_instances_split[split]

        if self.train_unseen_task:
            num_tasks = self.num_test_tasks
            tasks = self.precomputed_tasks[-num_tasks:]
        else:
            if split == 'train':
                num_tasks = self.num_train_tasks
                tasks = self.precomputed_tasks[:num_tasks]
            else:
                num_tasks = self.num_test_tasks
                tasks = self.precomputed_tasks[-num_tasks:]

        if self.train_unseen_task:

            task_id = self.eval_task_id
            task = tasks[task_id]
            # Note: Even though train and test task id's overlap, they index into different tables
            assert task_id < self.num_test_tasks
            sentences, labels, lengths = self.construct_data(
                rng, task_id, task, num_instances)

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
        else:
            dataset_map = OrderedDict()
            for i in range(num_tasks):
                task_id = i
                task = tasks[i]
                sentences, labels, lengths = self.construct_data(
                    rng, task_id, task, num_instances)
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
            self.datasets[split] = MultiCorpusSampledDataset(
                dataset_map, num_samples=self.sample_num_tasks)

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

    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['split_data'] = split_data

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']

        #sample_size = sample['target'].size(0)
        sample_size = 1

        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        self.logging_diagnostics = outputs.keys()

        for diagnostic in outputs:
            value = outputs[diagnostic]
            if type(value) == torch.Tensor:
                value = value.item()
            logging_output[diagnostic] = value

        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        optimizer.zero_grad()
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
        sample['net_input']['mode'] = 'eval'
        if 'meta' in model.training_mode:
            if self.batch_version:
                sample['net_input']['num_tasks'] = self.sample_num_tasks
            # Eval mode: Use 25% of the data to validation. The 75% is used for training by meta-learned
            # models and ignored by non-meta learning models.
            with torch.set_grad_enabled(True):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        agg_logging_outputs = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        for other_metrics in self.logging_diagnostics:
            agg_logging_outputs[other_metrics] = sum(
                log[other_metrics] for log in logging_outputs if other_metrics in log
            )
        return agg_logging_outputs
