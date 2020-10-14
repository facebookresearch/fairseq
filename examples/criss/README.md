# Cross-lingual Retrieval for Iterative Self-Supervised Training

https://arxiv.org/pdf/2006.09526.pdf

## Introduction

CRISS is a multilingual sequence-to-sequnce pretraining method where mining and training processes are applied iteratively, improving cross-lingual alignment and translation ability at the same time.

## Unsupervised Machine Translation
##### 1. Download and decompress CRISS checkpoints
```
cd examples/criss
wget https://dl.fbaipublicfiles.com/fairseq/models/criss/criss_checkpoints.tar.gz
tar -xf criss_checkpoints.tar.gz
```
##### 2. Download and preprocess Flores test dataset
```
bash download_and_preprocess_flores_test.sh
```

##### 3. Run Evaluation on Sinhala-English
```
bash unsupervised_mt/eval.sh
```

## Sentence Retrieval
##### 1. Download and preprocess Tatoeba dataset
```
bash download_and_preprocess_tatoeba.sh
```

##### 2. Run Sentence Retrieval on Tatoeba Kazakh-English
```
bash sentence_retrieval/sentence_retrieval_tatoeba.sh
```

## Mining
##### 1. Mine pseudo-parallel
```
bash sentence_retrieval/sentence_retrieval_tatoeba.sh
```

## Citation
```bibtex
@article{tran2020cross,
  title={Cross-lingual retrieval for iterative self-supervised training},
  author={Tran, Chau and Tang, Yuqing and Li, Xian and Gu, Jiatao},
  journal={arXiv preprint arXiv:2006.09526},
  year={2020}
}
```
