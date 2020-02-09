from unittest import TestCase

import fairseq
from fairseq.optim import optimizer_registry

from torch import nn

class TestOptimizer(TestCase):

    def setUp(self) -> None:
        self.model = self.DummyModel()

    def tearDown(self) -> None:
        self.model = None

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Linear(10, 10)

        def forward(self, *input, **kwargs):
            return self.f(input)

    def _test_optimizer(self, name, cls, *args, **kwargs):
        optim = optimizer_registry.get(name, params=self.model.parameters(), *args, **kwargs)

        self.assertIsInstance(optim, cls)

    def test_adafactor(self):
        self._test_optimizer("adafactor", fairseq.optim.adafactor.FairseqAdafactor, lr=[1e-3])

    def test_adadelta(self):
        self._test_optimizer("adadelta", fairseq.optim.adadelta.Adadelta, lr=[1e-3])

    def test_adagrad(self):
        self._test_optimizer("adagrad", fairseq.optim.adagrad.Adagrad, lr=[1e-3])

    def test_adam(self):
        self._test_optimizer("adam", fairseq.optim.adam.FairseqAdam, lr=[1e-3])
        
    def test_adamax(self):
        self._test_optimizer("adamax", fairseq.optim.adamax.FairseqAdamax, lr=[1e-3])

    def test_lamb(self):
        self._test_optimizer("lamb", fairseq.optim.fused_lamb.FairseqLAMB, lr=[1e-3])

    def test_nag(self):
        self._test_optimizer("nag", fairseq.optim.nag.FairseqNAG, lr=[1e-3])

    def test_sgd(self):
        self._test_optimizer("sgd", fairseq.optim.sgd.SGD, lr=[1e-3])