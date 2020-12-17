import unittest
from agg_results import make_sweep_table


class TestAggResults(unittest.TestCase):
    def test_make_sweep_table(self):
        try:
            df = make_sweep_table(
                "fb_sweep/mock_results/*/*.log", log_pattern="valid", interactive=True
            )
            assert "valid_ppl" in df.columns
        except ImportError:
            pass
