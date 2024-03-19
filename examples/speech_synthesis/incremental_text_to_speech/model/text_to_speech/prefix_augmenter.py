import torch
import logging
import random

logger = logging.getLogger(__name__)


class PartialSequenceAugmenter(object):
    def __init__(self, src_dict):
        super(PartialSequenceAugmenter, self).__init__()
        self.src_dict = src_dict

    def augment(self, sample, prefix_augment_ratio=3):
        device = sample["net_input"]["src_tokens"].device

        bsz, max_src_len = sample["net_input"]["src_tokens"].size()
        _, max_feature_len, feature_dim = sample["target"].size()
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        durations = sample["durations"]

        pitches = sample["pitches"]
        energies = sample["energies"]
        target = sample["target"]

        prefix_lengths = (src_lengths / prefix_augment_ratio).long() * random.randrange(1, prefix_augment_ratio)
        max_prefix_len = prefix_lengths.max()
        prefix_token_mask = torch.arange(max_src_len).to(device).view(1, max_src_len).expand(bsz, -1) >= \
                            prefix_lengths.view(bsz, 1).expand(bsz, -1)

        src_tokens = src_tokens.masked_fill_(prefix_token_mask, self.src_dict.pad())[:, :max_prefix_len]
        durations = durations.masked_fill_(prefix_token_mask, 0)[:, :max_prefix_len]
        pitches = pitches.masked_fill_(prefix_token_mask, 0)[:, :max_prefix_len]
        energies = energies.masked_fill_(prefix_token_mask, 0)[:, :max_prefix_len]

        prefix_feature_lengths = durations.sum(dim=-1)
        max_prefix_feature_len = prefix_feature_lengths.max()
        prefix_feature_mask = torch.arange(max_feature_len).to(device).view(1, max_feature_len).expand(bsz, -1) >= \
                              prefix_feature_lengths.view(bsz, 1).expand(bsz, -1)
        target = target.masked_fill_(prefix_feature_mask.unsqueeze(-1), 0)[:, :max_prefix_feature_len, :]

        return {
            "id": sample["id"],
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": prefix_lengths,
                "prev_output_tokens": None,
            },
            "speaker": sample["speaker"],
            "target": target,
            "durations": durations,
            "pitches": pitches,
            "energies": energies,
            "target_lengths": prefix_feature_lengths,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "src_texts": sample["src_texts"],
        }
