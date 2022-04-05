### NormFormer
This is the code for the ["NormFormer: Improved Transformer Pretraining with Extra Normalization"](https://arxiv.org/abs/2110.09456)
- 2021-10-19: Commands for CLM Experiments
- Coming soon: Commands for MLM experiments

If you have any issues or questions please post a github issue and tag `@sshleifer`.


### Data
- To preprocess language modeling data, see [here](https://github.com/pytorch/fairseq/blob/d0fbcb0baef6f6ff3425ded62d8daea0e8b12114/examples/language_model/README.md#1-preprocess-the-data).
- The replication commands below expect `$DATA` to be the path to the binarized data directory.
- Note that NormFormer results in Table 2 use a much larger private dataset, and to get good results you should adapt the pre-processing instructions to your dataset and compare to a baseline on the same data, rather than Table 2.
- The code uses `FSDP`, which requires `pip install fairscale>=0.4.0`.


### Modify existing Command
To modify an existing `fairseq-train` command to use NormFormer, simply add the following flags:
```bash
fairseq-train  ... \
    --scale-attn --scale-fc --scale-heads
```
- you probably also want to increase your learning rate
- if your model is small, you may want to add `--scale-resids`

### Exact Training Commands

- Note that NormFormer results in Table 2 use a much larger private dataset, and to get good results you should adapt the pre-processing instructions to your dataset.
The full commands are functions defined here, so to run them you must `source examples/normformer/train_lm.sh`.
- We default `--distributed-world-size 8`. You should adjust `--update-freq` and `--batch-size` and such that the effective batch size is (1024x1024x0.5) tokens for 125M and 355M,
    and (1024x1024) for 1.3B parameter and above. For small models, `--update-freq`=256/`global_bs`. For large models, `--update-freq`=512/`global_bs`, where `global_bs` = `--batch-size` * `--distributed-world-size`
- The small models will all train on as few as 8 GPUs.

```bash
train_125M --lr 6e-4  # GPT-3 Replicated
train_125M --lr 1e-3  # stronger high-lr baseline
train_125M --lr 3e-3 --scale-attn --scale-fc --scale-heads # No scale-resids
train_125M --lr 3e-3 --scale-attn --scale-fc --scale-heads --scale-resids  # Best command
```

```bash
train_355M --lr 6e-4  # GPT-3 Replicated
train_355M --lr 1e-3  # stronger high-lr baseline
train_355M --lr 1e-3 --scale-attn --scale-fc --scale-heads # No scale-resids
train_355M --lr 1e-3 --scale-attn --scale-fc --scale-heads --scale-resids  # Slightly better
```

```bash
train_1.3B --lr 2e-4  # GPT-3 Replicated
train_1.3B --lr 6e-4  # stronger high-lr baseline
train_1.3B --lr 6e-4 --scale-attn --scale-fc --scale-heads # NormFormer
```

```bash
train_2.7B --lr 1.6e-4  # GPT-3 Replicated
train_2.7B --lr 1.6e-4 --activation-fn relu_squared # stronger Relu^2 baseline
train_2.7B --lr 6e-4 --activation-fn relu_squared --scale-attn --scale-fc --scale-heads # NormFormer 2.7B
```


### Citation
```bibtex
@misc{shleifer2021normformer,
      title={NormFormer: Improved Transformer Pretraining with Extra Normalization},
      author={Sam Shleifer and Jason Weston and Myle Ott},
      year={2021},
      eprint={2110.09456},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
