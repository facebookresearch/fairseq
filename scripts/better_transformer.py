# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

import click
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from fairseq.models.transformer import TransformerConfig as FairseqTransformerConfig
from fairseq.models.transformer import TransformerEncoder as FairseqTransformerEncoder

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def numerical_test(lengths, truth_tensors, test_list):
    """
    truth_tensors is the source of truth.
    test_dict looks like
    [
        (name, out_tensors, atol, rtol),
        ...
    ]
    """
    for name, out_tensors, rtol, atol in test_list:
        n_failures = 0
        max_diff = 0
        for (length, truth, out) in zip(lengths, truth_tensors, out_tensors):
            cut_truth = truth[:length]
            cut_out = out[:length]
            max_diff = max(max_diff, torch.max(torch.abs(cut_truth - cut_out)))
            if not torch.allclose(cut_truth, cut_out, atol=atol, rtol=rtol):
                n_failures += 1

        if n_failures == 0:
            print(f"{name} PASS")
        else:
            print(f"{name} FAIL {n_failures}/{len(lengths)}. Max diff is {max_diff}")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--save", is_flag=True, default=False)
@click.option("--load", is_flag=True, default=False)
@click.option("--half", is_flag=True, default=False)
@click.option("--bt2fairseq", is_flag=True, default=False)
def transformer(
    save,
    load,
    half,
    bt2fairseq,
):
    xlarge = False
    large = False
    DEFAULT_PADDING_IDX = 1
    avg_sequence_length = 128
    max_sequence_length = 256
    batch_size = 64

    class FairseqEncoder(torch.nn.Module):
        def __init__(
            self,
            embed_dim,
            attention_heads,
            ffn_embed_dim,
            num_layers,
            embedding_layer,  # torch.nn.Embedding. Must have a padding_idx field
            dropout=0,
            normalize_before=False,
            torch_encoder=None,  # torch encoder that you can map weights from
            activation="relu",
        ):
            super().__init__()

            cfg = FairseqTransformerConfig()
            cfg.encoder.embed_dim = embed_dim
            cfg.encoder.attention_heads = attention_heads
            cfg.encoder.ffn_embed_dim = ffn_embed_dim
            cfg.dropout = dropout
            cfg.encoder.normalize_before = normalize_before
            cfg.encoder.layers = num_layers
            # make embedding behavior same as other encoders
            cfg.no_token_positional_embeddings = True
            cfg.no_scale_embedding = True
            cfg.activation_fn = activation
            dictionary = {}  # TODO: verify what this is

            self.encoder = FairseqTransformerEncoder(
                cfg, dictionary, embedding_layer, return_fc=False
            )

            if torch_encoder is not None:
                for src_layer, dst_layer in zip(
                    torch_encoder.layers, self.encoder.layers
                ):
                    w_q, w_k, w_v = src_layer.self_attn.in_proj_weight.chunk(3, dim=0)
                    b_q, b_k, b_v = src_layer.self_attn.in_proj_bias.chunk(3, dim=0)

                    dst_layer.self_attn.q_proj.weight = torch.nn.Parameter(w_q)
                    dst_layer.self_attn.q_proj.bias = torch.nn.Parameter(b_q)
                    dst_layer.self_attn.k_proj.weight = torch.nn.Parameter(w_k)
                    dst_layer.self_attn.k_proj.bias = torch.nn.Parameter(b_k)
                    dst_layer.self_attn.v_proj.weight = torch.nn.Parameter(w_v)
                    dst_layer.self_attn.v_proj.bias = torch.nn.Parameter(b_v)

                    dst_layer.self_attn.out_proj.weight = (
                        src_layer.self_attn.out_proj.weight
                    )
                    dst_layer.self_attn.out_proj.bias = (
                        src_layer.self_attn.out_proj.bias
                    )

                    dst_layer.fc1.weight = src_layer.linear1.weight
                    dst_layer.fc1.bias = src_layer.linear1.bias

                    # fairseq may use fusedlayernorm from nvidia apex - diff properties
                    dst_layer.self_attn_layer_norm.load_state_dict(
                        src_layer.norm1.state_dict()
                    )

                    dst_layer.fc2.weight = src_layer.linear2.weight
                    dst_layer.fc2.bias = src_layer.linear2.bias

                    dst_layer.final_layer_norm.load_state_dict(
                        src_layer.norm2.state_dict()
                    )

            # self.encoder = self.encoder.eval().cuda().half()

        def forward(self, tokens, src_lengths=None):
            return self.encoder(
                tokens,
                src_lengths=src_lengths,
                return_all_hiddens=False,
                token_embeddings=None,
            )

    def get_layers_embedding_dim_num_heads_for_configuration(xlarge, large):
        if xlarge:
            # XLM-R extra large (no BERT-XL exists)
            L = 24  # Layers
            D = 2560  # Embedding Dim
            H = 32  # Number of Heads
            FD = 10240  # Feed-forward network dim
            V = 30000  # Vocab Size
        elif large:
            # BERT-large
            L = 24
            D = 1024
            H = 16
            FD = 4096
            V = 30000
        else:
            # BERT-base
            L = 12
            D = 768
            H = 12
            FD = 3072
            V = 30000

        return (L, D, H, FD, V)

    # Better transformer
    class PTTransformer(torch.nn.Module):
        def __init__(self, transformer, embedding):
            super().__init__()
            self.transformer = transformer
            self.embedding = embedding
            self.padding_idx = DEFAULT_PADDING_IDX

        def forward(self, x):
            padding_mask = None
            if not x.is_nested:
                padding_mask = x.eq(self.padding_idx)
            x = self.embedding(x)
            return self.transformer(x, src_key_padding_mask=padding_mask)

    def make_transformer():
        return (
            PTTransformer(
                torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=D,
                        nhead=H,
                        dim_feedforward=FD,
                        batch_first=True,
                        activation="relu",
                    ),
                    num_layers=L,
                    enable_nested_tensor=False,
                ),
                embedding_layer,
            )
            .eval()
            .cuda()
        )

    def copy_weights(layers_fairseq, layers_bt):
        for src_layer, dst_layer in zip(layers_fairseq, layers_bt):
            w_q = src_layer.self_attn.q_proj.weight
            b_q = src_layer.self_attn.q_proj.bias
            w_k = src_layer.self_attn.k_proj.weight
            b_k = src_layer.self_attn.k_proj.bias
            w_v = src_layer.self_attn.v_proj.weight
            b_v = src_layer.self_attn.v_proj.bias
            dst_layer.self_attn.in_proj_weight = torch.nn.Parameter(
                torch.cat((w_q, w_k, w_v), dim=0)
            )
            dst_layer.self_attn.in_proj_bias = torch.nn.Parameter(
                torch.cat((b_q, b_k, b_v), dim=0)
            )

            dst_layer.self_attn.out_proj.weight = src_layer.self_attn.out_proj.weight
            dst_layer.self_attn.out_proj.bias = src_layer.self_attn.out_proj.bias

            dst_layer.linear1.weight = src_layer.fc1.weight
            dst_layer.linear1.bias = src_layer.fc1.bias
            dst_layer.linear2.weight = src_layer.fc2.weight
            dst_layer.linear2.bias = src_layer.fc2.bias

            dst_layer.norm1.weight = src_layer.self_attn_layer_norm.weight
            dst_layer.norm1.bias = src_layer.self_attn_layer_norm.bias
            dst_layer.norm2.weight = src_layer.final_layer_norm.weight
            dst_layer.norm2.bias = src_layer.final_layer_norm.bias

    (L, D, H, FD, V) = get_layers_embedding_dim_num_heads_for_configuration(
        xlarge, large
    )
    embedding_layer = torch.nn.Embedding(V, D, DEFAULT_PADDING_IDX)
    # True means BT as source and fairseq is target, False means the other way
    # mode1 = False
    if bt2fairseq:
        # BT as source and fairseq is target, copy BT's weight to fairseq
        transformer = make_transformer()
        fairseq_transformer = (
            FairseqEncoder(
                D,
                H,
                FD,
                L,
                embedding_layer,
                dropout=0,
                normalize_before=False,
                torch_encoder=transformer.transformer,
                activation="relu",
            )
            .eval()
            .cuda()
        )
        if half:
            transformer.half()
            fairseq_transformer.half()
    if not bt2fairseq:
        # the other way around, fairseq is source and BT is target,copy fairseq's weight to BT
        transformer = make_transformer()
        fairseq_transformer = (
            FairseqEncoder(
                D,
                H,
                FD,
                L,
                embedding_layer,
                dropout=0,
                normalize_before=False,
                torch_encoder=None,
                activation="relu",
            )
            .eval()
            .cuda()
        )
        # for the test where we need to load existing ckpt. It is tested that after loading
        # the ckpt, the results between fairseq_transformer(BT kernel) equals BT
        if half:
            transformer.half()
            fairseq_transformer.half()
        if save:
            torch.save(fairseq_transformer.state_dict(), "./fairseq.pt")
            sys.exit(0)
        if load:
            fairseq_transformer.load_state_dict(torch.load("./fairseq.pt"))
        # copy
        copy_weights(fairseq_transformer.encoder.layers, transformer.transformer.layers)

    device = "cuda"
    lengths = (avg_sequence_length,) * batch_size
    tokens = torch.full(
        (batch_size, max_sequence_length),
        DEFAULT_PADDING_IDX,
        device=device,
        dtype=torch.long,
    )
    for i in range(batch_size):
        tokens[i, : lengths[i]] = torch.randint(
            DEFAULT_PADDING_IDX + 1,
            V - 1,
            size=(lengths[i],),
            device=device,
            dtype=torch.long,
        )
    # mask
    if half:
        lengths_tensor = torch.Tensor(lengths).cuda().half()
    else:
        lengths_tensor = torch.Tensor(lengths).cuda()

    with torch.inference_mode():
        fs_output = fairseq_transformer(tokens, lengths_tensor)["encoder_out"][0]
        fs_output = fs_output.transpose(0, 1)
    with torch.inference_mode():
        t_output = transformer(tokens)
    test_lst = [
        # (name, output, relative tolerance, absolute tolerance)
        ("FS", fs_output, 1e-4, 9e-3),
    ]
    numerical_test(lengths, t_output, test_lst)

    iters = 100
    t = benchmark_torch_function(iters, transformer, tokens)

    def bert_flops(B, T, D, L):
        mlp = 2 * (B * T * D * 4 * D) + 2 * (B * T * D * 4 * D)
        qkv = 3 * 2 * B * T * D * D
        attn = 2 * B * D * T * T + 2 * B * D * T * T + 2 * B * T * D * D
        return L * (mlp + qkv + attn)

    flops = bert_flops(batch_size, avg_sequence_length, D, L)
    flops_e = (
        FlopCountAnalysis(transformer, (tokens[:, :avg_sequence_length])).total() * 2
    )
    with torch.inference_mode():
        bt = benchmark_torch_function(iters, transformer, tokens)
        fst = benchmark_torch_function(
            iters, fairseq_transformer, tokens, lengths_tensor
        )

        def metrics(tt, baseline=None):
            if baseline:
                return metrics(tt) + f", Speedup: {baseline / tt:.2f}x"
            return f"{tt * 1.0e3:.2f} ms/iter, {flops_e / tt / 1.0e12:.2f} TFLOP/s"

        results = [
            f"Seed: {seed}",
            f"Padded tokens: {(1-sum(lengths)/(tokens.numel()))*100:.2f}%",
            f"Batch shape: {tokens.shape}",
            f"Analytical flops per batch: {flops/ batch_size / 1e9:.2f} GFLOPS",
            f"Empirical flops per batch: {flops_e/ batch_size / 1e9:.2f} GFLOPS",
            f"B: {batch_size}",
            f"T: {avg_sequence_length}",
            f"TMax: {max_sequence_length}",
            f"Eager Time: {metrics(t)}",
            f"BetterTransformer: {metrics(bt, t)}",
            f"FST: {metrics(fst, t)}",
        ]
        print("===========Speedup Results")
        print("; ".join(results))


if __name__ == "__main__":
    cli()
