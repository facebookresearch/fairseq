from examples.speech_recognition.modules.specaugment import SpecAugment
from .asr_test_base import get_dummy_input
import torch
import unittest

class SpecaugmentTest(unittest.TestCase):

    def setUp(self):
        self.forward_input = get_dummy_input()

    def test_forward(self):
        batch = {'net_input': self.forward_input,
                 'id': torch.arange(self.forward_input['src_tokens'].size(0))
                 }
        new_batch = batch.copy()
        sa = SpecAugment(13, 13, 2, 2, 1.0)
        new_batch = sa(new_batch)
        #Verify that not spectrogram values are not changed
        for k in batch:
            if k != "net_input":
                for a, b in zip(batch[k], new_batch[k]):
                    self.assertEqual(a, b)

        for k in batch['net_input']:
            #Verify that not spectrogram values are not changed
            if k != 'src_tokens':
                for a, b in zip(batch['net_input'][k].view(-1), new_batch['net_input'][k].view(-1)):
                    self.assertEqual(a, b)
            else:
                for n in range(batch['net_input'][k].size(0)):
                    # Size stays the same
                    self.assertEqual(batch['net_input'][k][n].size(), new_batch['net_input'][k][n].size())
                    # At least a column is made of only zeros
                    self.assertFalse(new_batch['net_input'][k][n].sum(0).bool().all())
                    # At least a row is made of only zeros
                    self.assertFalse(new_batch['net_input'][k][n].sum(1).bool().all())
                    # The values did not change or are zeros
                    for a, b in zip(batch['net_input'][k][n].view(-1), new_batch['net_input'][k][n].view(-1)):
                        self.assertTrue(b == 0.0 or a == b)


if __name__ == '__main__':
    unittest.main()
