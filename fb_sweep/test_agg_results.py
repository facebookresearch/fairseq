import unittest
from agg_results import make_sweep_table, find_common_prefix


class TestAggResults(unittest.TestCase):
    def test_make_sweep_table(self):
        test_data_glob = "fb_sweep/mock_results/*/*.log"
        df = make_sweep_table(test_data_glob, log_pattern="valid", interactive=True)
        # these next two checks break if without removing common prefix from keys
        assert "valid_ppl" not in df.columns
        assert "ppl" in df.columns

        train_inner_df = make_sweep_table(
            test_data_glob, log_pattern="train_inner", interactive=True
        )
        assert train_inner_df.ppl.notnull().all()

        train_df = make_sweep_table(
            test_data_glob, log_pattern="train", interactive=True
        )
        assert train_df.ppl.notnull().all()
        assert train_df.shape[1] == train_inner_df.shape[1] == 14

    def test_find_common_prefix(self):
        assert find_common_prefix([]) == ""
        assert find_common_prefix(["train_wall"]) == ""
        assert find_common_prefix(["train_wall", "train_ppl"]) == "train_"
        assert find_common_prefix(["train_wall", "train_ppl", "wall"]) == ""
        assert (
            find_common_prefix(["train_wall", "train_ppl", "train_train_wall"])
            == "train_"
        )
