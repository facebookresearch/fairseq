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

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.


# Usage
This clone of fairseq supports ```Knowledge Distillation```, ```Recurrent Stacking```, and ```Adapter tuning``` for ```Transformers``` and ```translation``` task. You can add the following flags to ```fairseq-train``` to use them:

- **Knowledge Distillation**: The original implementation was sourced from [LeslieOverfitting](https://github.com/LeslieOverfitting/selective_distillation). The total loss for KD is given by $\mathcal{L}_{total} = \alpha \mathcal{L}_{KD} + (1 - \alpha) \mathcal{L}_{NLL}$, where $\mathcal{L}_{KD}$ is the CrossEntropy between the student and teacher probability distributions at each timestep, and $\mathcal{L}_{NLL}$ is the Negative Log-likelihood loss between the prediction and target. Setting $\alpha = 1$, will yield pure Word-Level Distillation ([Hinton _et al_.](https://arxiv.org/abs/1503.02531), [Kim & Rush](https://aclanthology.org/D16-1139))


```
--task translation_with_kd \
--kd-strategy word_seq_level \
--teacher-checkpoint-path $teacher_ckpt \
--criterion label_smoothed_cross_entropy_with_kd \
--alpha 0.5
```

The training with Batch-Level and Global-Level KD ([Wang _et al_.](https://aclanthology.org/2021.acl-long.504)) can be done as follows:
```
--task translation_with_kd \
--kd-strategy batch_level \
--teacher-checkpoint-path $teacher_ckpt \
--criterion label_smoothed_cross_entropy_with_kd \
--kd-rate 0.5
```
and 
```
--task translation_with_kd \
--kd-strategy global_level \
--teacher-checkpoint-path $teacher_ckpt \
--criterion label_smoothed_cross_entropy_with_kd \
--kd-rate 0.5 \
--kd-queue-size 1000000
```
Lastly, the Global-Language-wise selection approach ([Gumma _et al_.](https://arxiv.org/abs/2304.09388)) can used by:
```
--task translation_with_kd \
--kd-strategy global_language_wise \
--teacher-checkpoint-path $teacher_ckpt \
--criterion label_smoothed_cross_entropy_with_kd \
--kd-rate 0.5 \
--kd-queue-size 1000000
```
Here, similar to Global-Level KD, each language has it's own Global FIFO queue, which makes it suitable for multilingual KD with imabalanced datasets. This technique requires language tags to be added to each translation pair similar to [Ramesh _et al_.](https://aclanthology.org/2022.tacl-1.9/). These tags will help the model break the batch into respective languages an d push them into the corresponding Global language queues. Note that, each FIFO language queue, irrespective of language abundance will be of same size, i.e. ```$kd_queue_sz```. I know this does not sound so good, and I am working on an alternative.


- **Recurrent Stacking** ([Dabre & Fujita](https://ojs.aaai.org/index.php/AAAI/article/view/4590)): RS is an extreme parameter sharing technique in which all the layers in the encoder/decoder are shared. That is, implementation wise, only one layer exists in the module and rest $N-1$ are mere references to it. RS can be activated with the following flags:
```
--encoder-recurrent-stacking 6 \
--decoder-recurrent-stacking 6
```

- **Adapter Tuning** ([Houlsby _et al_.](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf), [Bapna & Firat](https://aclanthology.org/D19-1165/)): Small FFN blocks with a bottleneck hidden layer are added in the Transformer layer for additional parameterization. As of now, the adapter hidden layer supports ReLU, GELU, SiLU and Tanh activations. Note that, all other parameters except these adapters will be frozen during training. You can also you gating for the skip-connection inside the adapter using the flag ```--encoder-adapter-use-gating``` or ```--decoder-adapter-use-gating```. The adapters can be added and trained using following flags:

```
--encoder-add-adapters \
--encoder-adapter-reduction-factor 2 \
--encoder-adapter-ids as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
--encoder-train-adapter hi \
--decoder-add-adapters \
--decoder-adapter-reduction-factor 2 \
--decoder-adapter-ids as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
--decoder-train-adapter hi \
--adapter-activation-fn gelu \
--load-checkpoint-liberally
```

During evaluation, you can add the following flags to ```fairseq-interactive```
```
--activate-encoder-adapter hi \
--activate-decoder-adapter hi
```
to use that specific adapter (```hi``` in this case).

_Additional Information_: [Mundra _et al_.](https://arxiv.org/pdf/2305.07491.pdf) is a neat comprehensive analysis paper about the efficiency of Adapters, and good place to begin if you want to dive into Adapter tuning.


- **Miscellaneous**:

  - _Factorized Embedding Parameterization_ ([Lan _et al_.](https://iclr.cc/virtual_2020/poster_H1eA7AEtvS.html)): Similar to ALBERT, the large embeddings can be parameterized by adding an intermediate bottleneck layer, i.e. the instead of being a single $|V| \times d_m$ matrix, the Embedding consists of two pieces of sizes $|V| \times k$ and $k \times d_m$ respectively, where $k < d_m$. This helps curb the number of parameters in the the Embedding layer, which can one of the most bulky components. Factorized embeddings as can used as:
  ```
  --encoder-factorized-embed-dim 64 \
  --decoder-factorized-embed-dim 64
  ```

  - _Sanity Validation steps_: Similar to Pytorch-Lightning Trainer, a full pass over the validation set can be run in the beginning of training to smoke out any bugs in the training/validation. It can be activated with the flag:
  ```
  --run-sanity-validation-steps
  ```

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

fairseq(-py) is MIT-licensed.
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

I will try my best to keep this repo sync'ed with upstream [fairseq](https://github.com/facebookresearch/fairseq) repository. This clone is very dynamic and can have broken stuff once in a while. So feel free to raise any issues or pull-requests to clear any bugs or introduce new features. BTW, I am trying to integrate ```torch.compile``` with ```fairseq```, and I get flooded with a bunch of warning and logging messages from ```torch._dynamo``` and ```torch._inductor```, so any leads on this will be really helpful!