# Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)

This page includes instructions for reproducing results from the paper [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](https://arxiv.org/abs/1902.07816).

## Training a new model on WMT'17 En-De

First, follow the [instructions to download and preprocess the WMT'17 En-De dataset](../translation#prepare-wmt14en2desh).
Make sure to learn a joint vocabulary by passing the `--joined-dictionary` option to `fairseq-preprocess`.

Then we can train a mixture of experts model using the `translation_moe` task.
Use the `--method` option to choose the MoE variant; we support hard mixtures with a learned or uniform prior (`--method hMoElp` and `hMoEup`, respectively) and soft mixures (`--method sMoElp` and `sMoEup`).

To train a hard mixture of experts model with a learned prior (`hMoElp`) on 1 GPU:
```
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wmt17_en_de \
  --max-update 100000 \
  --task translation_moe \
  --method hMoElp --mean-pool-gating-network \
  --num-experts 3 \
  --arch transformer_vaswani_wmt_en_de --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0007 --min-lr 1e-09 \
  --dropout 0.1 --weight-decay 0.0 --criterion cross_entropy \
  --max-tokens 3584 \
  --update-freq 8
```

**Note**: the above command assumes 1 GPU, but accumulates gradients from 8 fwd/bwd passes to simulate training on 8 GPUs.
You can accelerate training on up to 8 GPUs by adjusting the `CUDA_VISIBLE_DEVICES` and `--update-freq` options accordingly.

Once a model is trained, we can generate translations from different experts using the `--gen-expert` option.
For example, to generate from expert 0:
```
$ fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/checkpoint_best.pt 
    --beam 1 --remove-bpe \
    --task translation_moe \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 3 \
    --gen-expert 0 \
```

You can also use `scripts/score_moe.py` to compute pairwise BLEU and average oracle BLEU.
We'll first download a tokenized version of the multi-reference WMT'14 En-De dataset:
```
$ wget dl.fbaipublicfiles.com/fairseq/data/wmt14-en-de.extra_refs.tok
```

Next apply BPE on the fly and run generation for each expert:
```
$ BPEROOT=examples/translation/subword-nmt/
$ BPE_CODE=examples/translation/wmt17_en_de/code
$ for EXPERT in $(seq 0 2); do \
    cat wmt14-en-de.extra_refs.tok | grep ^S | cut -f 2 | \
      python $BPEROOT/apply_bpe.py -c $BPE_CODE | \
      fairseq-interactive data-bin/wmt17_en_de \
        --path checkpoints/checkpoint_best.pt \
        --beam 1 --remove-bpe \
        --buffer 500 --max-tokens 6000 ; \
        --task translation_moe \
        --method hMoElp --mean-pool-gating-network \
        --num-experts 3 \
        --gen-expert $EXPERT \
  done > wmt14-en-de.extra_refs.tok.gen.3experts
```

Finally compute pairwise BLUE and average oracle BLEU:
```
$ python scripts/score_moe.py --sys wmt14-en-de.extra_refs.tok.gen.3experts --ref wmt14-en-de.extra_refs.tok
pairwise BLEU: 48.26
avg oracle BLEU: 49.50
#refs covered: 2.11
```

This reproduces row 3 from Table 7 in the paper.

## Citation

```bibtex
@article{shen2019mixture,
  title = {Mixture Models for Diverse Machine Translation: Tricks of the Trade},
  author = {Tianxiao Shen and Myle Ott and Michael Auli and Marc'Aurelio Ranzato},
  journal = {arXiv preprint arXiv:1902.07816},
  year = 2019,
}
```
