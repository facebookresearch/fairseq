# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    TokenBlockDataset,
)
from fairseq.tasks import FairseqTask, register_task


@register_task('review_task')
class ReviewTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of tokens per sample')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>) to each sample')

        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--num_train', default=10000, type=int,
                            help='Num training examples')
        parser.add_argument('--num_test', default=10000, type=int,
                            help='Num test examples')
        parser.add_argument('--vocab_size', default=10, type=int,
                            help='Vocabulary size')
        parser.add_argument('--num_train_tasks', default=5, type=int,
                            help='Number of training tasks')
        parser.add_argument('--num_test_tasks', default=5, type=int,
                            help='Number of test tasks')
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
        parser.add_argument('--load_tasks_file', default='/checkpoint/llajan/tasks.txt', type=str,
                            help='Tasks file.')
        parser.add_argument('--all_task_examples', action='store_true',
                            help='Feed all task training examples as input.')
        parser.add_argument('--no_training', action='store_true',
                            help='No fine-tuning.')

        # fmt: on

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.no_training = args.no_training

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0

        vocab = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # e.g., /path/to/data/train.{bin,idx}
        split_path = os.path.join(data_path, split)
        dataset = data_utils.load_indexed_dataset(split_path, self.vocab, self.args.dataset_impl)
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # view the dataset as a single 1D tensor and chunk into samples
        # according to break_mode
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            block_size=self.args.tokens_per_sample,
            pad=self.vocab.pad(),
            eos=self.vocab.eos(),
            # "complete_doc" mode splits the underlying dataset into documents
            # (where documents are separated by blank lines) and then returns
            # chunks of sentences up to block_size. A single document may span
            # multiple chunks, but each chunk will only contain sentences from
            # a single document.
            break_mode='complete_doc',
        )

        # prepend a beginning of sentence token (<s>) to each sample
        if self.args.add_bos_token:
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # we predict the next word, so roll the inputs by 1 position
        input_tokens = RollDataset(dataset, shifts=1)
        target_tokens = dataset

        # define the structure of each batch. "net_input" is passed to the
        # model's forward, while the full sample (including "target") is passed
        # to the criterion
        dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': RightPadDataset(
                        input_tokens,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    'src_lengths': NumelDataset(input_tokens, reduce=False),
                },
                'target': RightPadDataset(
                    target_tokens,
                    pad_idx=self.target_dictionary.pad(),
                ),
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(target_tokens, reduce=True),
            },
            sizes=[input_tokens.sizes],
        )

        # shuffle the dataset and then sort by size
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))
            dataset = SortDataset(
                dataset,
                sort_order=[
                    shuffle,
                    input_tokens.sizes,
                ],
            )

        self.datasets[split] = dataset

    def _get_loss(self, sample, model, criterion, split_data=False):

        targets = sample['target']
        sample['net_input']['targets'] = targets
        sample['net_input']['split_data'] = split_data

        outputs = model(**sample['net_input'])

        loss = outputs['post_loss_train']

        # sample_size = sample['target'].size(0)
        # sample_size = 1
        sample_size = sample['nsentences']

        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample['nsentences'],
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
        self.sample_num_tasks = 8
        loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
