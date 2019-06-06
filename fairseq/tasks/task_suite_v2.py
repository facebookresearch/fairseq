import torch

from fairseq.tasks import register_task
from fairseq.tasks.task_suite_base import TaskSuiteBase


@register_task('task_suite_v2')
class TaskSuite_v2(TaskSuiteBase):

    @classmethod
    def setup_task(cls, args, **kwargs):
        return TaskSuite_v2(args)

    def __init__(self, args):
        super().__init__(args)
        self.output_vocab_size = self.num_classes

    def construct_data(self, task_id, examples):

        sentences, lengths, labels = [], [], []

        for instance in examples:

            sentence, label = instance
            sentence = [task_id, self.cls_encode] + self.input_vocab.encode(sentence)
            sentences.append(torch.LongTensor(sentence))
            lengths.append(self.max_seq_len)
            labels.append(torch.LongTensor([label]))

        return sentences, labels, lengths
