# Using xFormers with FairSeq

[xFormers](https://github.com/facebookresearch/xformers) is a xFormers is a modular library for flexibly generating transformer architectures with interoperable and optimized building blocks.
The current integration allows for FairSeq users to use an attention variant available in the xFormers repository.

In order to enable xFormers, all that needs to be passed in is a string representing an [xFormers attention config](https://github.com/facebookresearch/xformers/blob/5f754129bfb1ea53747b1ab2077261ea762faa47/xformers/components/attention/base.py#L18).

The various attention variants can be found [here](https://github.com/facebookresearch/xformers/tree/main/xformers/components/attention).
These include sparse attention and blocksparse attention.

For example, you could pass in the following args:
 ```python
decoder_xformers_att_config = '{"name": "scaled_dot_product"}'

encoder_xformers_att_config = '{"name": "linformer", "seq_len": "256"}'
 ```

In order to use blocksparse attention you would have to additionally pass in a blocksparse layout and blocksize. For example:

 ```python

  xformers_att_config = '{"name": "scaled_dot_product"}'
  xformers_blocksparse_blocksize = 16
  xformers_blocksparse_layout = torch.ones(
      seq_len // xformers_blocksparse_blocksize,
      seq_len // xformers_blocksparse_blocksize,
  )

 xf_blocksparse_mha = (
        MultiheadAttention(
            embedding,
            num_heads,
            dropout=0.0,
            add_zero_attn=add_zero_attn,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=xformers_blocksparse_layout,
            xformers_blocksparse_blocksize=xformers_blocksparse_blocksize,
        )

 ```

The xFormers repository currenlty has benchmarks on the [runtime](https://github.com/facebookresearch/xformers/blob/main/docs/plots/runtime_vs_attention.png)
and [memory usage](https://github.com/facebookresearch/xformers/blob/main/docs/plots/memory_vs_attention.png) of the various attentions.
