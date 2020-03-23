# Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2019)
This page contains information for how to train models with Quantization Noise.

Check out our blog post [here](link_to_blog_post) and read the paper [here](link_to_paper).

Looking for pretrained models? They will be added shortly.
Looking for vision models? Check out this repository [here](link_to_vision_repo)

## Citation:
To be added shortly

## Example usage

Training a model with Quant-Noise improves the performance in subsequent inference-time quantization.

To train a model with Quant-Noise, add the following flags:
```
--quant-noise 0.1 --quant-noise-block-size 8
```
We recommend training with 0.05 to 0.2 Quant-Noise, a value that worked well in our experiments. For the block-size, we recommend training with block-size 8.

Quant-Noise can also be combined with LayerDrop (see [here](https://github.com/pytorch/fairseq/tree/master/examples/layerdrop)) to add its pruning effect to the quantized model and make the model even smaller. See that readme for additional details. We recommend training with LayerDrop 0.1 or 0.2.

Looking to reproduce the NLP results in the paper?

1. To train RoBERTa, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
2. To train Language Models on Wikitext-103, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/language_model)

Looking to reproduce the Vision results in the paper?

Check out this [repository](add_link_here) to reproduce our vision models.

## Having an issue or have a question?

Please open an issue in this repository with the details of your question. Thanks!
