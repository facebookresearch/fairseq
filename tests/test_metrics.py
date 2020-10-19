# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

from fairseq import metrics


class TestMetrics(unittest.TestCase):
    def test_nesting(self):
        with metrics.aggregate() as a:
            metrics.log_scalar("loss", 1)
            with metrics.aggregate() as b:
                metrics.log_scalar("loss", 2)

        self.assertEqual(a.get_smoothed_values()["loss"], 1.5)
        self.assertEqual(b.get_smoothed_values()["loss"], 2)

    def test_new_root(self):
        with metrics.aggregate() as a:
            metrics.log_scalar("loss", 1)
            with metrics.aggregate(new_root=True) as b:
                metrics.log_scalar("loss", 2)

        self.assertEqual(a.get_smoothed_values()["loss"], 1)
        self.assertEqual(b.get_smoothed_values()["loss"], 2)

    def test_nested_new_root(self):
        with metrics.aggregate() as layer1:
            metrics.log_scalar("loss", 1)
            with metrics.aggregate(new_root=True) as layer2:
                metrics.log_scalar("loss", 2)
                with metrics.aggregate() as layer3:
                    metrics.log_scalar("loss", 3)
                    with metrics.aggregate(new_root=True) as layer4:
                        metrics.log_scalar("loss", 4)
            metrics.log_scalar("loss", 1.5)

        self.assertEqual(layer4.get_smoothed_values()["loss"], 4)
        self.assertEqual(layer3.get_smoothed_values()["loss"], 3)
        self.assertEqual(layer2.get_smoothed_values()["loss"], 2.5)
        self.assertEqual(layer1.get_smoothed_values()["loss"], 1.25)

    def test_named(self):
        name = str(uuid.uuid4())
        metrics.reset_meters(name)

        with metrics.aggregate(name):
            metrics.log_scalar("loss", 1)

        metrics.log_scalar("loss", 3)

        with metrics.aggregate(name):
            metrics.log_scalar("loss", 2)

        self.assertEqual(metrics.get_smoothed_values(name)["loss"], 1.5)

    def test_nested_duplicate_names(self):
        name = str(uuid.uuid4())
        metrics.reset_meters(name)

        with metrics.aggregate(name):
            metrics.log_scalar("loss", 1)
            with metrics.aggregate() as other:
                with metrics.aggregate(name):
                    metrics.log_scalar("loss", 2)
            metrics.log_scalar("loss", 6)

        self.assertEqual(metrics.get_smoothed_values(name)["loss"], 3)
        self.assertEqual(other.get_smoothed_values()["loss"], 2)


if __name__ == "__main__":
    unittest.main()
