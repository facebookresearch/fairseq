# Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)

This page includes instructions for reproducing results from the paper [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](https://arxiv.org/abs/1902.07816).

## Download data

First, follow the [instructions to download and preprocess the WMT'17 En-De dataset](../translation#prepare-wmt14en2desh).
Make sure to learn a joint vocabulary by passing the `--joined-dictionary` option to `fairseq-preprocess`.

## Train a model

Then we can train a mixture of experts model using the `translation_moe` task.
Use the `--method` flag to choose the MoE variant; we support hard mixtures with a learned or uniform prior (`--method hMoElp` and `hMoEup`, respectively) and soft mixures (`--method sMoElp` and `sMoEup`).
The model is trained with online responsibility assignment and shared parameterization.

The following command will train a `hMoElp` model with `3` experts:
```bash
fairseq-train --ddp-backend='no_c10d' \
    data-bin/wmt17_en_de \
    --max-update 100000 \
    --task translation_moe --user-dir examples/translation_moe/src \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 3 \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 \
    --dropout 0.1 --weight-decay 0.0 --criterion cross_entropy \
    --max-tokens 3584
```

## Translate

Once a model is trained, we can generate translations from different experts using the `--gen-expert` option.
For example, to generate from expert 0:
```bash
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/checkpoint_best.pt \
    --beam 1 --remove-bpe \
    --task translation_moe --user-dir examples/translation_moe/src \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 3 \
    --gen-expert 0
```

## Evaluate

First download a tokenized version of the WMT'14 En-De test set with multiple references:
```bash
wget dl.fbaipublicfiles.com/fairseq/data/wmt14-en-de.extra_refs.tok
```

Next apply BPE on the fly and run generation for each expert:
```bash
BPE_CODE=examples/translation/wmt17_en_de/code
for EXPERT in $(seq 0 2); do \
    cat wmt14-en-de.extra_refs.tok \
    | grep ^S | cut -f 2 \
    | fairseq-interactive data-bin/wmt17_en_de \
        --path checkpoints/checkpoint_best.pt \
        --beam 1 \
        --bpe subword_nmt --bpe-codes $BPE_CODE \
        --buffer-size 500 --max-tokens 6000 \
        --task translation_moe --user-dir examples/translation_moe/src \
        --method hMoElp --mean-pool-gating-network \
        --num-experts 3 \
        --gen-expert $EXPERT ; \
done > wmt14-en-de.extra_refs.tok.gen.3experts
```

Finally use `score_moe.py` to compute pairwise BLUE and average oracle BLEU:
```bash
python examples/translation_moe/score.py --sys wmt14-en-de.extra_refs.tok.gen.3experts --ref wmt14-en-de.extra_refs.tok
# pairwise BLEU: 48.26
# #refs covered: 2.11
# multi-reference BLEU (leave-one-out): 59.46
```
This matches row 3 from Table 7 in the paper.

## Citation

```bibtex
@article{shen2019mixture,
  title = {Mixture Models for Diverse Machine Translation: Tricks of the Trade},
  author = {Tianxiao Shen and Myle Ott and Michael Auli and Marc'Aurelio Ranzato},
  journal = {International Conference on Machine Learning},
  year = 2019,
}
```
