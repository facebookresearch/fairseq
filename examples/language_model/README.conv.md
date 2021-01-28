# Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)

## Example usage

First download and preprocess the data following the main [language modeling README](README.md).

Then to train a convolutional LM using the `fconv_lm_dauphin_wikitext103`
architecture:
```bash
fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/fconv_wikitext-103 \
    --arch fconv_lm_dauphin_wikitext103 \
    --adaptive-softmax-cutoff 10000,20000,200000 \
    --dropout 0.2 \
    --criterion adaptive_loss \
    --optimizer nag --clip-norm 0.1 --weight-decay 5e-06 \
    --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --max-tokens 1024 --tokens-per-sample 1024 \
    --ddp-backend legacy_ddp \
    --max-epoch 35
```

And evaluate with:
```bash
fairseq-eval-lm data-bin/wikitext-103 --path checkpoints/fconv_wiki103/checkpoint_best.pt
```

## Citation

```bibtex
@inproceedings{dauphin2017language,
  title={Language Modeling with Gated Convolutional Networks},
  author={Dauphin, Yann N and Fan, Angela and Auli, Michael and Grangier, David},
  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
  pages={933--941},
  year={2017},
  organization={JMLR}
}
```
