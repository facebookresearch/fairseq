import unittest

import torch
from examples.speech_recognition.modules.time_stretch import TimeStretch, time_stretch_seq

class TimeStretchTest(unittest.TestCase):

    def test_singlesentencelength(self):
        input = torch.Tensor([[0.38, 0.56, 0.98, -0.12, -0.67, 1.3],
                 [-0.62, -0.74, 0.21, 1.73, -0.45, -0.54],
                 [-0.5, -0.83, 1.67, -1.78, 0.35, -0.12]]*100)
        stretched = time_stretch_seq(input, 10, low=1.0, high=1.0)
        self.assertEqual(len(input), len(stretched))
        stretched = time_stretch_seq(input, 10, low=0.8, high=0.8)
        self.assertEqual(len(input)*0.8, len(stretched))
        stretched = time_stretch_seq(input, 10, low=1.2, high=1.2)
        self.assertEqual(len(input)*1.2, len(stretched))

    def test_singlesentencelengthw15(self):
        input = torch.Tensor([[0.38, 0.56, 0.98, -0.12, -0.67, 1.3],
                 [-0.62, -0.74, 0.21, 1.73, -0.45, -0.54],
                 [-0.5, -0.83, 1.67, -1.78, 0.35, -0.12]]*100)
        w=15
        stretched = time_stretch_seq(input, w, low=1.0, high=1.0)
        self.assertEqual(len(input), len(stretched))
        stretched = time_stretch_seq(input, w, low=0.8, high=0.8)
        self.assertEqual(len(input)*0.8, len(stretched))
        stretched = time_stretch_seq(input, w, low=1.2, high=1.2)
        self.assertEqual(len(input)*1.2, len(stretched))

    def test_singlesentencelengthw19(self):
        input = torch.Tensor([[0.38, 0.56, 0.98, -0.12, -0.67, 1.3],
                 [-0.62, -0.74, 0.21, 1.73, -0.45, -0.54],
                 [-0.5, -0.83, 1.67, -1.78, 0.35, -0.12]]*100)
        w=19
        stretched = time_stretch_seq(input, w, low=1.0, high=1.0)
        self.assertEqual(len(input), len(stretched))


    def test_singlesentencecontent(self):
        input = torch.Tensor([[0.38, 0.56, 0.98, -0.12, -0.67, 1.3],
                              [-0.62, -0.74, 0.21, 1.73, -0.45, -0.54],
                              [-0.5, -0.83, 1.67, -1.78, 0.35, -0.12]] * 3)
        stretched = time_stretch_seq(input, 3, low=2.0, high=2.0)
        self.assertEqual(0.0, sum(sum(input[0:3,:]-stretched[0:6:2,:])))

    def test_batch(self):
        unit = [[0.38, 0.56, 0.98, -0.12, -0.67, 1.3],
               [-0.62, -0.74, 0.21, 1.73, -0.45, -0.54],
               [-0.5, -0.83, 1.67, -1.78, 0.35, -0.12]]
        input = torch.zeros((3, 330, 6))
        input[0, :300, :] = torch.Tensor(unit * 100)
        input[1, :240, :] = torch.Tensor(unit * 80)
        input[2, :330, :] = torch.Tensor(unit * 110)
        batch = {'id': [0, 1, 2],
                 'nsentences': 3,
                 'ntokens': 300 + 240 + 330,
                 'net_input': {
                     'src_tokens': input,
                     'src_lengths': [300, 240, 330],
                     'prev_output_tokens': None,
                 },
                 'target': [0,1,2,3,4],
            }

        new_batch = TimeStretch(10, low=1.0, high=1.0)(batch)

        for i in range(len(batch['id'])):
            self.assertEqual(batch['id'][i], new_batch['id'][i])
        self.assertEqual(batch['nsentences'], new_batch['nsentences'])
        self.assertEqual(batch['ntokens'], new_batch['ntokens'])
        self.assertEqual(batch['target'], new_batch['target'])
        self.assertEqual(0.0, sum(sum(sum(batch['net_input']['src_tokens'] - new_batch['net_input']['src_tokens']))))
