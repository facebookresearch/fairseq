# Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)
This page contains information for how to train models with LayerDrop.

Looking for pretrained models? They will be added shortly.

Looking for code for other forms of Structured Dropout? It will be added shortly.

## Citation:
```bibtex
@article{fan2019reducing,
  title={Reducing Transformer Depth on Demand with Structured Dropout},
  author={Fan, Angela and Grave, Edouard and Joulin, Armand},
  journal={arXiv preprint arXiv:1909.11556},
  year={2019}
}
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
```
python eval_lm.py /path/to/wikitext-103 --path '/path/to/model/checkpoint' --model-overrides "{'decoder_layers_to_keep':'0,2,4,6,8,10,12,14'}"
```
This model override command overrides the training parameters and updates the model arguments so that the pruned model is run instead of the full model.


Looking to reproduce the results in the paper?

1. For Translation on WMT en-de, we followed this setting [here](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md)
2. To train RoBERTa, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
3. To train Language Models on Wikitext-103, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/language_model)


## Tips

1. If you would like to train large models with better performance, LayerDrop should be set to a smaller value such as 0.1 or 0.2. Too much LayerDrop will mean the model has too much regularization, so may not reach the best performance. Since LayerDrop adds regularization, you may achieve the best performance by slightly reducing the amount of standard dropout (for example, reduce by 0.1).

2. If you would like to train large models to be pruned and made smaller, LayerDrop should be set to a larger value such as 0.5 if you want to prune very aggressively (such as removing half the network or more). If you would like to prune fewer layers away, LayerDrop can be set to a smaller value such as 0.2.

3. When pruning layers at inference time, it is best to spread out the layers remaining so they are evenly spaced throughout the network. For example, if you want to remove 50% of the network, keeping every other layer is good.

## Having an issue or have a question?

Please open an issue in this repository with the details of your question. Thanks!
