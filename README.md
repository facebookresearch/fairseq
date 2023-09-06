<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://opensource.fb.com/support-ukraine"><img alt="Support Ukraine" src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" /></a>
  <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
  <a href="https://app.circleci.com/pipelines/github/facebookresearch/fairseq/"><img alt="CicleCI Status" src="https://circleci.com/gh/facebookresearch/fairseq.svg?style=shield" /></a>
</p>

--------------------------------------------------------------------------------

`Fairseq(-py)` is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.


# Usage
This clone of fairseq supports `Knowledge Distillation`, `Recurrent Stacking`, and `Adapter tuning` for the `Transformer` model and the `translation` task. You can add the following flags to `fairseq-train` to use them:

- **Knowledge Distillation**: The original implementation was sourced from [LeslieOverfitting](https://github.com/LeslieOverfitting/selective_distillation) and [MANGA-UOFA](https://github.com/MANGA-UOFA/fdistill)

  - Pure Word-Level Distillation ([Hinton _et al_.](https://arxiv.org/abs/1503.02531)) can be achieved by: 
    - `--task translation_with_kd --kd-strategy word_level --teacher-checkpoint-path $teacher_ckpt --criterion label_smoothed_cross_entropy_with_kd `
    - Note that, there no NLL Loss between the gold targets and predictions. The only loss is the KL-Divergence between the student and teacher distributions ($`\mathcal{L}`$ = $`\mathcal{L}_{KD}`$)

  - [Kim & Rush](https://aclanthology.org/D16-1139) extend this idea and add a NLL Loss between the predictions and target and modify the loss as $`\mathcal{L}`$ = $`\mathcal{L}_{KD}`$ + $`\mathcal{L}_{NLL}`$. The same can be achieved with the following flags:
    - `--task translation_with_kd --kd-strategy word_seq_level --teacher-checkpoint-path $teacher_ckpt --criterion label_smoothed_cross_entropy_with_kd `

  - Training with Batch-Level and Global-Level KD ([Wang _et al_.](https://aclanthology.org/2021.acl-long.504)) can be done as follows:
    - `--task translation_with_kd --kd-strategy batch_level --teacher-checkpoint-path $teacher_ckpt --criterion label_smoothed_cross_entropy_with_kd --kd-rate $kd_rate`
    - `--task translation_with_kd --kd-strategy global_level --teacher-checkpoint-path $teacher_ckpt --criterion label_smoothed_cross_entropy_with_kd --kd-rate $kd_rate --kd-queue-size $kd_queue_sz`

  - Lastly, the Global-Language-wise selection approach ([Gumma _et al_.](https://aclanthology.org/2023.eamt-1.11/)) can used by:
    - `--task translation_with_kd --kd-strategy global_language_wise --teacher-checkpoint-path $teacher_ckpt --criterion label_smoothed_cross_entropy_with_kd --kd-rate $kd_rate --kd-queue-size $kd_queue_sz --kd-language-tags $language_tags` (note that the `$language_tags` should be a comma separated string of language tags)

  - Here, similar to Global-Level KD, each language has its own Global FIFO queue, which makes it suitable for multilingual KD with imbalanced datasets. This technique requires adding language tags to each translation pair, similar to [Ramesh _et al_.](https://aclanthology.org/2022.tacl-1.9/). These tags will help the model break the batch into respective languages and push them into the corresponding Global language queues. Note that each FIFO language queue, irrespective of language abundance, will be of the same size, i.e., ```$kd_queue_sz```. I know this does not sound so good, and I am working on an alternative.

  - *UPDATE-1*: _Initially, the KD Loss was implemented as the CrossEntropy between student and teacher model distributions, but it was very unustable in mixed-precision training, and led to `inf` loss. Hence, the latest implementation uses KL-Divergence, which is much more stable and easy to compute in PyTorch_.
  - *UPDATE-2*: _Based on [Wen _et al_.](https://aclanthology.org/2023.acl-long.605.pdf), newer variants for KD Loss have been implemented, wiz. `js_div` and `tvd`. They can be used by setting the flag `--kd-criterion $kd_criterion`. By default, `kl_div` is used._


- **Recurrent Stacking** ([Dabre & Fujita](https://ojs.aaai.org/index.php/AAAI/article/view/4590)): RS is an extreme parameter sharing technique in which all the layers in the encoder/decoder are shared. Implementation-wise, only one layer exists in the module, and the rest $N-1$ are mere references to it. RS can be activated with the following flags: `--encoder-recurrent-stacking $encoder_recurrent_stacking --decoder-recurrent-stacking $decoder_recurrent_stacking`

- **Adapter Tuning** ([Houlsby _et al_.](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf), [Bapna & Firat](https://aclanthology.org/D19-1165/)): Small FFN blocks with a bottleneck hidden layer are added in the Transformer layer for additional parameterization. The adapter hidden layer currently supports `ReLU`, `GELU`, `SiLU`, and `Tanh` activations. Note that all other parameters except these adapters will be frozen during training. Gating for the skip-connection inside the adapter can be enabled using the flags `--encoder-adapter-use-gating` or `--decoder-adapter-use-gating`. 

  - The adapters can be added and trained using the following flags: `--encoder-add-adapters --encoder-adapter-reduction-factor $encoder_reduction_factor --encoder-adapter-ids $encoder_adapter_ids_list --encoder-train-adapter $encoder_train_adapter_id --decoder-add-adapters --decoder-adapter-reduction-factor $decoder_reduction_factor --decoder-adapter-ids $decoder_adapter_ids_list --decoder-train-adapter $decoder_train_adapter_id --adapter-activation-fn $adapter_activation_fn --load-checkpoint-liberally`
  - During evaluation, you can add the following flags to `fairseq-interactive`  to use that specific adapter `--activate-encoder-adapter $encoder_adapter_id --activate-decoder-adapter $decoder_adapter_id`. Here `$encoder_adapter_ids_list`/`$decoder_adapter_ids_list` is a comma separated list like `as,bn,gu,hi,kn,ml,mr,or,pa,ta,te` and `$encoder_train_adapter_id`/`$decoder_train_adapter_id` is one of the adapters from that list (say `hi`) which will get trained. Essentially, the above flags add a block of bottleneck adapters and they can be trained one-after the another.


- **Miscellaneous**:
  - _Factorized Embedding Parameterization_ ([Lan _et al_.](https://iclr.cc/virtual_2020/poster_H1eA7AEtvS.html)): Similar to ALBERT, the large embeddings can be parameterized by adding an intermediate bottleneck layer, i.e., the instead of being a single $|V| \times d_m$ matrix, the Embedding consists of two pieces of sizes $|V| \times k$ and $k \times d_m$ respectively, where $k < d_m$. This helps curb the number of parameters in the Embedding layer, which can one of the most bulky components. Factorized embeddings can be used as:`--encoder-factorized-embed-dim $encoder_fac_embed_dim --decoder-factorized-embed-dim $decoder_fac_embed_dim `. A non-linear activation function can be applied to the intermediate bottleneck layer specifying it in the flag `--factorized-embed-activation-fn $fac_embed_activation_fn`.

  - When using a penultimate linear transformation before the final projection on to the vocabulary, an activation can be added by `--decoder-output-activation-fn $decoder_out_activation_fn`

  - _Sanity Validation steps_: Similar to `Pytorch-Lightning Trainer`, a full pass over the validation set can be run at the beginning of training to eliminate any bugs in the training/validation. It can be activated with the flag: `--run-sanity-validation-steps`

  - Added support for `Python 3.11` and dumped the version from `0.12.2` -> `0.12.3.1`


# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

`fairseq(-py)` is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

# Final Note

I will try my best to keep this repo sync'ed with upstream [fairseq](https://github.com/facebookresearch/fairseq) repository. This clone is very dynamic and can have broken stuff once in a while. So feel free to raise any issues or pull-requests to clear any bugs or introduce new features.
