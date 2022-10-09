# Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)
This page contains information for how to train models with LayerDrop, based on this [paper](https://arxiv.org/abs/1909.11556).

## Citation:
If you found this technique useful, please cite our paper:
```bibtex
@article{fan2019reducing,
  title={Reducing Transformer Depth on Demand with Structured Dropout},
  author={Fan, Angela and Grave, Edouard and Joulin, Armand},
  journal={arXiv preprint arXiv:1909.11556},
  year={2019}
}
```

## Pre-trained models

Model | Description | Download
---|---|---
`layerdrop_wmt_en_de_12_6` | Transformer + LayerDrop 0.2 trained on WMT16 en-de with 12 encoder and 6 decoder layers | [layerdrop_wmt_en_de_12_6.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/layerdrop_wmt_en_de_12_6.tar.gz)
`roberta_layerdrop.base` | RoBERTa Base + LayerDrop 0.2 | [roberta_layerdrop.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta_layerdrop.base.qnli.tar.gz)
`roberta_layerdrop.large` | RoBERTa Large + LayerDrop 0.2 | [roberta_layerdrop.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta_layerdrop.large.tar.gz)
`roberta_layerdrop.large.mnli` | `roberta_layerdrop.large` finetuned on [MNLI](http://www.nyu.edu/projects/bowman/multinli) | [roberta_layerdrop.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta_layerdrop.large.mnli.tar.gz)
`roberta_layerdrop.large.qnli` | `roberta_layerdrop.large` finetuned on [QNLI](https://arxiv.org/abs/1804.07461) | [roberta_layerdrop.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta_layerdrop.large.qnli.tar.gz)


Evaluate performance of these pre-trained models:
```bash
# Example for Machine Translation
fairseq-generate /path/to/bped/wmt/data --path nmt_checkpoint.pt \
  --beam 8 --lenpen 0.4 \
  --batch-size 64 \
  --remove-bpe \
  --gen-subset test > wmt16_gen.txt
bash scripts/compound_split_bleu.sh wmt16_gen.txt
# prints BLEU4 = 30.17
```

```python
# Example for RoBERTa + LayerDrop finetuned on MNLI:
from fairseq.models.roberta import RobertaModel

roberta_layerdrop = RobertaModel.from_pretrained(
    '/path/to/MNLI/model',
    checkpoint_file='mnli_checkpoint.pt',
    data_name_or_path='/path/to/MNLI/data/MNLI-bin'
)
label_map = {0: 'contradiction', 2: 'neutral', 1: 'entailment'}
ncorrect, nsamples = 0, 0
roberta_layerdrop.cuda()
roberta_layerdrop.eval()
with open('/path/to/MNLI/data/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = roberta_layerdrop.encode(sent1, sent2)
        prediction = roberta_layerdrop.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# prints | Accuracy:  0.9026999490575649


# Example for RoBERTa + LayerDrop finetuned on QNLI:
roberta = RobertaModel.from_pretrained(
    '/path/to/QNLI/model',
    checkpoint_file='qnli_checkpoint.pt',
    data_name_or_path='/path/to/QNLI/data/QNLI-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.target_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('/path/to/QNLI/data/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# prints | Accuracy:  0.9480139117700896
```


## Example usage

To train a model with LayerDrop, add the following flags. We recommend 0.2, a value that worked well in our experiments. For Language Models that are decoder-only, you need only the decoder flag. For RoBERTa, an encoder, you need only the encoder flag. The encoder and decoder LayerDrop values can be set differently.
```
--encoder-layerdrop 0.2 --decoder-layerdrop 0.2
```

To prune a model that has been trained with LayerDrop, add the following flags followed by a comma separated list of which layers you would like to keep.
```
--encoder-layers-to-keep 0,2,4,6,8,10,12,14 --decoder-layers-to-keep 0,2,4,6,8,10,12,14
```
Setting these flags should print a message such as:
```
| Pruning model to specified layer configuration
```
You should also see a smaller number of parameters in the model, for example the 16-Layer Transformer Language Model prints:
```
num. model params: 246933504
```
while a model pruned to 8 Layers prints:
```
num. model params: 146163712
```

If you would like to pick up training with a model that has been pruned, simply adding these flags is sufficient. If you would like to use a script that only does evaluation (no training), you may need to pass an override command. A specific example would be for language modeling:
```bash
fairseq-eval-lm /path/to/wikitext-103 \
  --path /path/to/model/checkpoint.pt \
  --model-overrides "{'decoder_layers_to_keep':'0,2,4,6,8,10,12,14'}"
```
This model override command overrides the training parameters and updates the model arguments so that the pruned model is run instead of the full model.

## Reproduce Paper Results

Looking to reproduce the results in the paper?

1. For Translation on WMT16 en-de, we followed this setting [here](https://github.com/pytorch/fairseq/blob/main/examples/scaling_nmt/README.md)
2. To train RoBERTa, we followed this setting [here](https://github.com/pytorch/fairseq/tree/main/examples/roberta)
3. To train Language Models on Wikitext-103, we followed this setting [here](https://github.com/pytorch/fairseq/tree/main/examples/language_model)


## Tips

1. If you would like to train large models with better performance, LayerDrop should be set to a smaller value such as 0.1 or 0.2. Too much LayerDrop will mean the model has too much regularization, so may not reach the best performance. Since LayerDrop adds regularization, you may achieve the best performance by slightly reducing the amount of standard dropout (for example, reduce by 0.1).

2. If you would like to train large models to be pruned and made smaller, LayerDrop should be set to a larger value such as 0.5 if you want to prune very aggressively (such as removing half the network or more). If you would like to prune fewer layers away, LayerDrop can be set to a smaller value such as 0.2. Our experiments were conducted with low values of LayerDrop (such as 0.1 and 0.2), for reference.

3. When pruning layers at inference time, it is best to spread out the layers remaining so they are evenly spaced throughout the network. For example, if you want to remove 50% of the network, keeping every other layer is good.


## FAQ

1. How did the sharing layers experiment work? In an appendix (https://openreview.net/pdf?id=SylO2yStDr) we added an experiment on Wikitext-103 language modeling that combined LayerDrop with Weight Sharing. We shared chunks of 2 layers such that every other layer had shared weights. For example, if our network has layers 1 through 6, then layer 1 and 2 are shared, layer 3 and 4 are shared, and layer 5 and 6 are shared.

2. LayerDrop hasn't been helping in my setting? During training time, LayerDrop can help regularize your network. This is most important if your network is already overfitting - if your network is underfitting, it is possible LayerDrop is adding too much regularization. We recommend using smaller values (such as 0.1 or 0.2) and also decreasing the quantity of standard dropout (for example, reduce by 0.1).

3. Can you train a model without LayerDrop and finetune with LayerDrop (e.g. for BERT)? In our experiments, we did not see great performance. Models such as RoBERTa have trained for a long time in the pre-training setting, so only finetuning with LayerDrop for a few epochs on a downstream task such as MNLI does not achieve the robustness required for successful pruning.


## Having an issue or have a question?

Please open an issue in this repository with the details of your question. Thanks!
