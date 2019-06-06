# from pdb import set_trace as bp
import torch

from fairseq.tasks import register_task
from fairseq.tasks.task_suite_base import TaskSuiteBase


@register_task('task_suite_matching')
class TaskSuite_matching_ctx_meta(TaskSuiteBase):

    @classmethod
    def setup_task(cls, args, **kwargs):
        return TaskSuite_matching_ctx_meta(args)

    def __init__(self, args):
        super().__init__(args)
        self.output_vocab_size = 1

    def construct_data_train(self, task_id, examples):

        sentences, lengths, labels = [], [], []

        for instance in examples:

            sentence, label = instance
            sentence = [task_id] + self.input_vocab.encode(sentence) + [self.label_encode[label]]
            # sentence = self.input_vocab.encode(sentence)
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths

    def construct_data_test(self, task_id, examples, train_examples, split):

        if split == 'train':
            return self.construct_data_train(task_id, train_examples)

        sentences, lengths, labels = [], [], []

        for instance in examples:

            sentence, label = instance

            sentence_pairs = []
            sentence_pair_labels = []
            for train_example in train_examples:

                train_sentence, train_label = train_example

                # Note: Using label_map instead of label_encode
                sentence_pair = self.input_vocab.encode(
                    sentence + train_sentence + [self.label_map[train_label]])
                sentence_pairs.append(torch.LongTensor(sentence_pair))
                sentence_pair_labels.append(train_label)
            sentence_pair_labels.append(label)

            sentence_pairs = torch.cat(sentence_pairs, 0)
            sentences.append(sentence_pairs)
            labels.append(torch.LongTensor(sentence_pair_labels))
            lengths.append(sentence_pairs.shape[0])

        return sentences, labels, lengths

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, self.args.max_positions)

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

        if not self.no_training:
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        # We need gradient computation
        sample['net_input']['mode'] = 'eval'
        if self.batch_version:
            sample['net_input']['num_tasks'] = self.sample_num_tasks
        if 'meta' in model.training_mode:
            # Eval mode: Use 25% of the data to validation. The 75% is used for training by meta-learned
            # models and ignored by non-meta learning models.
            with torch.set_grad_enabled(True):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion, split_data=True)
        else:
            with torch.no_grad():
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)

        return loss, sample_size, logging_output
