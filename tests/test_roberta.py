# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import unittest
from typing import Any, Dict, Sequence

import fairseq
import fairseq.options
import fairseq.tasks
import torch
from tests.utils import dummy_dictionary

VOCAB_SIZE = 100


@fairseq.tasks.register_task("fake_task")
class FakeTask(fairseq.tasks.LegacyFairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = dummy_dictionary(VOCAB_SIZE - 4)
        assert len(self.dictionary) == VOCAB_SIZE

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


@functools.lru_cache()
def get_toy_model(
    device: str,
    architecture: str = "roberta_enc_dec",
    **extra_args: Any,
):
    assert device in ("gpu", "cpu")
    kwargs = {
        "arch": architecture,
        # Use characteristics dimensions
        "encoder_layers": 3,
        "encoder_embed_dim": 12,
        "encoder_ffn_embed_dim": 14,
        "encoder_attention_heads": 4,
        "decoder_layers": 3,
        "decoder_embed_dim": 12,
        "decoder_ffn_embed_dim": 14,
        "decoder_attention_heads": 4,
        # Disable dropout so we have comparable tests.
        "dropout": 0,
        "attention_dropout": 0,
        "activation_dropout": 0,
        "encoder_layerdrop": 0,
        # required args
        "tokens_per_sample": 256,
        "data": "/tmp/test_roberta",
    }
    kwargs.update(extra_args)
    fake_task = FakeTask(kwargs)
    args = fairseq.options.get_args(
        task="online_backtranslation",
        mono_langs="en,ro",
        valid_lang_pairs="en-ro",
        **kwargs,
    )
    torch.manual_seed(0)
    model = fake_task.build_model(args)
    if device == "gpu":
        model.cuda()
    return fake_task, model


def mk_sample(
    lang: str, device: str, tok: Sequence[int] = None, batch_size: int = 2
) -> Dict[str, Any]:
    assert device in ("gpu", "cpu")
    if not tok:
        if lang == "en":
            tok = [10, 11, 12, 13, 14, 15, 2]
        else:
            tok = [20, 21, 22, 23, 24, 25, 26, 27, 2]

    batch = torch.stack([torch.tensor(tok, dtype=torch.long)] * batch_size)
    if device == "gpu":
        batch = batch.cuda()
    sample = {
        "net_input": {
            "src_tokens": batch,
            "prev_output_tokens": batch,
            "src_lengths": torch.tensor(
                [len(tok)] * batch_size, dtype=torch.long, device=batch.device
            ),
        },
        "target": batch[:, 1:],
    }
    return sample


def cpu_gpu(fn):
    def helper(self):
        fn(self, "cpu")
        if torch.cuda.is_available():
            fn(self, "gpu")

    return helper


def architectures(fn):
    def helper(self):
        for arch in ["roberta_enc_dec", "transformer"]:
            fn(self, arch)

    return helper


class RobertaTest(unittest.TestCase):
    def assertTensorEqual(self, t1, t2, delta: float = 1e-6):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        if delta == 0.0:
            self.assertEqual(t1.ne(t2).long().sum(), 0)
        else:
            self.assertEqual(((t2 - t1).abs() > delta).long().sum(), 0)

    def assertSharing(self, model, link_groups: Sequence[Sequence[str]]):
        ids = {}
        for group in link_groups:
            group_ids = {name: id(params(model, name)) for name in group}
            shared_id = group_ids[group[0]]
            self.assertEqual(group_ids, {name: shared_id for name in group})
            self.assertNotIn(shared_id, ids)
            ids[shared_id] = group

    def test_roberta_shared_params(self):
        _, roberta = get_toy_model("cpu", architecture="roberta")
        self.assertSharing(
            roberta,
            [
                [
                    "encoder.sentence_encoder.embed_tokens.weight",
                    "encoder.lm_head.weight",
                ]
            ],
        )

        _, roberta = get_toy_model(
            "cpu", architecture="roberta", untie_weights_roberta=True
        )
        self.assertSharing(
            roberta,
            [
                ["encoder.sentence_encoder.embed_tokens.weight"],
                ["encoder.lm_head.weight"],
            ],
        )

    def test_roberta_enc_dec_shared_params(self):
        # 3 distinct embeddings
        _, enc_dec = get_toy_model("cpu", architecture="roberta_enc_dec")
        self.assertSharing(
            enc_dec,
            [
                ["encoder.embed_tokens.weight"],
                ["decoder.embed_tokens.weight"],
                ["decoder.output_projection.weight"],
            ],
        )

        # 2 distinct embeddings, one for encoder, one for decoder
        _, enc_dec = get_toy_model(
            "cpu", architecture="roberta_enc_dec", share_decoder_input_output_embed=True
        )
        self.assertSharing(
            enc_dec,
            [
                ["encoder.embed_tokens.weight"],
                [
                    "decoder.embed_tokens.weight",
                    "decoder.output_projection.weight",
                ],
            ],
        )

        # shared embeddings
        _, enc_dec = get_toy_model(
            "cpu", architecture="roberta_enc_dec", share_all_embeddings=True
        )
        self.assertSharing(
            enc_dec,
            [
                [
                    "encoder.embed_tokens.weight",
                    "decoder.embed_tokens.weight",
                    "decoder.output_projection.weight",
                ]
            ],
        )

    def test_roberta_max_positions_is_correctly_set(self):
        device = "cpu"
        task, model = get_toy_model(device)
        max_pos = model.max_decoder_positions()
        self.assertEqual(max_pos, 256)
        self.assertEqual(max_pos, model.decoder.max_positions())
        self.assertEqual(max_pos, model.encoder.max_positions())
        self.assertEqual(max_pos, model.encoder.embed_positions.max_positions)

        sentence = [31 for _ in range(max_pos)]
        sample = mk_sample("en", device, sentence, batch_size=1)
        self.assertEqual(list(sample["net_input"]["src_lengths"]), [max_pos])
        self.assertEqual(len(sample["net_input"]["src_tokens"][0]), max_pos)
        x, _ = model.forward(**sample["net_input"])
        self.assertEqual(x.shape, (1, max_pos, VOCAB_SIZE))

    @cpu_gpu
    def test_roberta_forward_backward(self, device: str):
        _, model = get_toy_model(device)
        sample = mk_sample("en", device)
        en_tokens = sample["net_input"]["src_tokens"]
        (bs, l) = en_tokens.shape
        # Forward
        logits, _ = model(**sample["net_input"])
        self.assertEqual(logits.shape, (bs, l, VOCAB_SIZE))

        # Backward
        loss = logits.sum()
        loss.backward()

    @cpu_gpu
    def test_roberta_forward_backward_bs1(self, device: str):
        _, model = get_toy_model(device)
        sample = mk_sample("en", device, batch_size=1)
        o, _ = model.forward(**sample["net_input"])
        loss = o.sum()
        sample2 = mk_sample("ro", device, batch_size=1)
        o, _ = model.forward(**sample2["net_input"])
        loss += o.sum()
        loss.backward()

    @cpu_gpu
    def test_roberta_batching(self, device: str):
        """
        Checks that the batch of size 2 give twice the same results than the batch of size 1.
        """
        _, model = get_toy_model(device)
        sample = mk_sample("en", device, batch_size=1)
        slen = sample["net_input"]["src_lengths"][0]
        sample2 = mk_sample("en", device, batch_size=2)
        with torch.no_grad():
            z = model.encoder.forward(
                sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]
            )
            z = z["encoder_out"][-1]
            logits, _ = model.forward(**sample["net_input"])

            z2 = model.encoder.forward(
                sample2["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]
            )
            z2 = z2["encoder_out"][-1]
            logits2, _ = model.forward(**sample2["net_input"])

        self.assertEqual(z.shape, (slen, 1, 12))
        self.assertEqual(z2.shape, (slen, 2, 12))
        self.assertTensorEqual(logits2[0], logits2[1])
        self.assertTensorEqual(logits[0], logits2[0])

    @cpu_gpu
    def test_roberta_incremental_decoder(self, device: str):
        """
        Checks that incremental decoding yields the same result than non incremental one.
        """
        task, model = get_toy_model(device)

        en_sample = mk_sample("en", device)
        en_tokens = en_sample["net_input"]["src_tokens"]
        ro_sample = mk_sample("ro", device)
        ro_tokens = ro_sample["net_input"]["src_tokens"]

        en_enc = model.encoder.forward(
            en_tokens, src_lengths=en_sample["net_input"]["src_lengths"]
        )
        (bs, tgt_len) = ro_tokens.shape

        # Decode without incremental state
        ro_dec, _ = model.decoder.forward(ro_tokens, encoder_out=en_enc)
        self.assertEqual(ro_dec.shape, (bs, tgt_len, VOCAB_SIZE))
        self.assertTensorEqual(ro_dec[0], ro_dec[1])

        # Decode with incremental state
        inc_state = {}
        ro_dec_inc = []
        for l in range(tgt_len):
            ro, _ = model.decoder.forward(
                ro_tokens[:, : l + 1], encoder_out=en_enc, incremental_state=inc_state
            )
            self.assertEqual(ro.shape, (bs, 1, VOCAB_SIZE))
            ro_dec_inc.append(ro)

        for l in range(tgt_len):
            # Intra-batch
            self.assertTensorEqual(ro_dec_inc[l][0], ro_dec_inc[l][1])
            # Incremental vs non-incremental
            self.assertTensorEqual(ro_dec_inc[l][:, 0], ro_dec[:, l])


def params(model, name):
    if "." not in name:
        return getattr(model, name)

    prefix, name = name.split(".", 1)
    return params(getattr(model, prefix), name)
