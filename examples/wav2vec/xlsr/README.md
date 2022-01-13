# XLS-R

XLS-R is a set of large-scale models for self-supervised cross-lingual speech representation learning based on wav2vec 2.0. It was pretrained on 128 languages and approximately 436K hours of unlabeled speech data. With finetuning, these models achieve state of the art performance in speech translation, speech recognition and language identification. We evaluate the model across multiple benchmarks such as CoVoST-2 for speech translation, BABEL / MLS / CommonVoice / VoxPopuli for automatic speech recognition, and VoxLingua107 for language identification as we llas VoxCeleb1 for speaker identification. More details about this work can be found in our [paper](https://arxiv.org/pdf/2111.09296.pdf) and download links can be found below.

Model | Link
|------|------
XLS-R 300M | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)
XLS-R 1B | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt)
XLS-R 2B | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt)

You can also download these models [here](https://huggingface.co/models?other=xls_r) and read more about it in the [blogpost](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) from Hugging Face.

## Speech Translation Finetuned Models

We multilingually finetune XLS-R models on [CoVoST 2](https://github.com/facebookresearch/covost), which has 21 
into-English and 15 out-of-English directions.

Model | Directions | Link
|------|------|------
XLS-R 300M | 21 langs &#8594; En | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_300m_21_en.pt)
XLS-R 300M | En &#8594; 15 langs | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_300m_en_15.pt)
XLS-R 1B | 21 langs &#8594; En | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_1b_21_en.pt)
XLS-R 1B | En &#8594; 15 langs | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_1b_en_15.pt)
XLS-R 2B | 21 langs &#8594; En | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_2b_21_en.pt)
XLS-R 2B | En &#8594; 15 langs | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_2b_en_15.pt)
XLS-R 2B | 21 langs &#8594; En + En &#8594; 15 langs | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xls_r_2b_22_16.pt)

## ASR Finetuning

You can refer the original wav2vec documentation on detailed instructions about how to finetune a pretrained model with CTC [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#fine-tune-a-pre-trained-model-with-ctc). Below is an example command and you can find the values for different hyperparameters to reproduce the results in our paper.

```shell script
$ fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/xlsr/config \
    --config-name finetune
```

For finetuning the 300M as well as 1B model, we use the same hyperparameter setting defined in `finetune.yaml`. We vary `optimization.max_update` as described in the below table and the `optimization.lr` is picked from the interval [2e-5, 3e-4] based on dev word error rate.

Benchmark | Total Number of Updates
|------|------
Babel | 26000
Common Voice | 13000
VoxPopuli | 50000
MLS 10h | 20000

For finetuning the 2B model, we make some additional changes for `finetune.yaml` . We use the fully_sharded `distributed_training.ddp_backend` provided by the [fairscale](https://github.com/facebookresearch/fairscale) library and and set `model.activation_checkpoint` to true. We also increase `dataset.max_tokens` to 2560000 and use a total effective batch size of 2560000*24. We sweep for the best `optimization.lr` within the interval [3e−6,3e−5] using dev error rate. For common voice dataset, we pick the `model.mask_prob` for different languages among {0.30, 0.40} based on best dev error rate.



## Citation

Please cite as:

``` bibtex
@article{babu2021xlsr,
      title={XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale}, 
      author={Arun Babu and Changhan Wang and Andros Tjandra and Kushal Lakhotia and Qiantong Xu and Naman Goyal and Kritika Singh and Patrick von Platen and Yatharth Saraf and Juan Pino and Alexei Baevski and Alexis Conneau and Michael Auli},
      year={2021},
      volume={abs/2111.09296},
      journal={arXiv},
}
```


