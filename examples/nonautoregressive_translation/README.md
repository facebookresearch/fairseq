# Non-autoregressive Neural Machine Translation (NAT)

This page mainly includes instructions for reproducing results from the paper
* [Levenshtein Transformer (Gu et al., 2019)](https://arxiv.org/abs/1905.11006).

We also provided our own implementations for several popular non-autoregressive-based models as reference:<br>
* [Non-Autoregressive Neural Machine Translation (Gu et al., 2017)](https://arxiv.org/abs/1711.02281)<br>
* [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)](https://arxiv.org/abs/1802.06901)<br>
* [Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)](https://arxiv.org/abs/1902.03249)<br>
* [Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)](https://arxiv.org/abs/1904.09324v2)

## Dataset

First, follow the [instructions to download and preprocess the WMT'14 En-De dataset](../translation#prepare-wmt14en2desh).
Make sure to learn a joint vocabulary by passing the `--joined-dictionary` option to `fairseq-preprocess`.

### Knowledge Distillation
Following [Gu et al. 2019](https://arxiv.org/abs/1905.11006), [knowledge distillation](https://arxiv.org/abs/1606.07947) from an autoregressive model can effectively simplify the training data distribution, which is sometimes essential for NAT-based models to learn good translations.
The easiest way of performing distillation is to follow the [instructions of training a standard transformer model](../translation) on the same data, and then decode the training set to produce a distillation dataset for NAT.

### Download
We also provided the preprocessed [original](http://dl.fbaipublicfiles.com/nat/original_dataset.zip) and [distillation](http://dl.fbaipublicfiles.com/nat/distill_dataset.zip) datasets. Please build the binarized dataset on your own.


## Train a model

Then we can train a nonautoregressive model using the `translation_lev` task and a new criterion `nat_loss`.
Use the `--noise` flag to specify the input noise used on the target sentences.
In default, we run the task for *Levenshtein Transformer*, with `--noise='random_delete'`. Full scripts to run other models can also be found [here](./scripts.md).

The following command will train a *Levenshtein Transformer* on the binarized dataset.

```bash
fairseq-train \
    data-bin/wmt14_en_de_distill \
    --save-dir checkpoints \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --save-interval-updates 10000 \
    --max-update 300000
```

## Translate

Once a model is trained, we can generate translations using an `iterative_refinement_generator` which will based on the model's initial output and iteratively read and greedily refine the translation until (1) the model predicts the same translations for two consecutive iterations; or (2) the generator reaches the maximum iterations (`--iter-decode-max-iter`). Use `--print-step` to check the actual # of iteration for each sentence.

For *Levenshtein Transformer*, it sometimes helps to apply a `--iter-decode-eos-penalty` (typically, 0~3) to penalize the model finishing generation too early and generating too short translations.


For example, to generate with `--iter-decode-max-iter=9`:
```bash
fairseq-generate \
    data-bin/wmt14_en_de_distill \
    --gen-subset test \
    --task translation_lev \
    --path checkpoints/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 400
```
In the end of the generation, we can see the tokenized BLEU score for the translation.


## Citation

```bibtex
@article{gu2019levenshtein,
  title={Levenshtein Transformer},
  author={Gu, Jiatao and Wang, Changhan and Zhao, Jake},
  journal={arXiv preprint arXiv:1905.11006},
  year={2019}
}
```
