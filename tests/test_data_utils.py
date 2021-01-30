# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from fairseq.data.data_utils_fast import batch_by_size_fn
from fairseq.data.data_utils_fast import batch_by_size_vec


class TestBatchBySize(unittest.TestCase):
    @classmethod
    def batch_by_size_baseline(
        cls,
        indices,
        num_tokens_vec,
        max_tokens,
        max_sentences,
        bsz_mult,
    ):
        """Simple, reliable and slow implementation of batch by size """
        batches = []
        start = 0
        while start < len(indices):
            for end in range(start + 1, len(indices) + 1):
                max_val = max(num_tokens_vec[pos] for pos in range(start, end))
                sent_count = end - start
                num_tokens = max_val * sent_count
                overflow = num_tokens > max_tokens > 0 or sent_count > max_sentences > 0
                terminate = overflow or end == len(indices)
                if overflow:
                    sent_count -= 1
                if terminate:
                    if sent_count > bsz_mult:
                        sent_count = sent_count - sent_count % bsz_mult
                    batches.append(indices[start : start + sent_count])
                    start = start + sent_count
                    break
        return batches

    @classmethod
    def _get_error_message(
        cls, max_sentences, max_tokens, bsz_mult, num_tokens_vec, validation, results
    ):
        return f"""Reference batch_by_size implementation should produce
                    same output as the baseline method.
                Params:
                max_sentences={max_sentences},
                max_tokens={max_tokens},
                bsz_mult={bsz_mult},
                num_tokens_vec={num_tokens_vec},
                expected_batches={validation},
                returned_batches={results}"""

    def _compare_results(
        self,
        indices_len,
        batch_by_size_impl,
        max_sentences,
        max_tokens,
        bsz_mult,
        num_tokens_vec,
    ):
        indices = np.array(list(range(indices_len)))
        validation = self.batch_by_size_baseline(
            indices,
            num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            bsz_mult=bsz_mult,
        )
        results = batch_by_size_impl(
            indices,
            num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            bsz_mult=bsz_mult,
        )
        error_msg = self._get_error_message(
            max_sentences, max_tokens, bsz_mult, num_tokens_vec, validation, results
        )
        self.assertEqual(len(validation), len(results), error_msg)
        for first, second in zip(validation, results):
            self.assertTrue(np.array_equal(first, second), error_msg)

    def _run_compare_with_baseline_sweep(self, batch_by_size_impl):
        """Compare reference batch_by_size implementation with batch_by_size_baseline
        across a dense grid of hyperparam values"""
        MAX_MAX_TOKENS = 10
        NUM_TOKENS_VECS_COUNT = 5
        for indices_len in [10, 11]:  # try odd and even len of indices
            for max_sentences in range(0, indices_len + 2):
                for max_tokens in range(0, MAX_MAX_TOKENS):
                    for bsz_mult in range(1, max(MAX_MAX_TOKENS, indices_len) + 2):
                        for _ in range(NUM_TOKENS_VECS_COUNT):
                            num_tokens_vec = np.random.randint(
                                0, max_tokens + 1, size=indices_len
                            )
                            self._compare_results(
                                indices_len,
                                batch_by_size_impl,
                                max_sentences,
                                max_tokens,
                                bsz_mult,
                                num_tokens_vec,
                            )


class TestBatchBySizeVec(TestBatchBySize):
    def test_compare_with_baseline(self):
        self._run_compare_with_baseline_sweep(batch_by_size_vec)


class TestBatchBySizeFn(TestBatchBySize):
    def test_compare_with_baseline(self):
        def batch_by_size_fn_wrapper(
            indices,
            num_tokens_vec,
            max_tokens,
            max_sentences,
            bsz_mult,
        ):
            def num_tokens_fn(idx):
                return num_tokens_vec[idx]

            return batch_by_size_fn(
                indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult
            )

        self._run_compare_with_baseline_sweep(batch_by_size_fn_wrapper)


if __name__ == "__main__":
    unittest.main()
