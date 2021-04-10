# LASER  Language-Agnostic SEntence Representations

LASER is a library to calculate and use multilingual sentence embeddings.

You can find more information about LASER and how to use it on the official [LASER repository](https://github.com/facebookresearch/LASER).

This folder contains source code for training LASER embeddings.


## Prepare data and configuration file

Binarize your data with fairseq, as described [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#data-pre-processing).

Create a json config file with this format:
```
{
  "src_vocab": "/path/to/spm.src.cvocab",
  "tgt_vocab": "/path/to/spm.tgt.cvocab",
  "train": [
    {
      "type": "translation",
      "id": 0,
      "src": "/path/to/srclang1-tgtlang0/train.srclang1",
      "tgt": "/path/to/srclang1-tgtlang0/train.tgtlang0"
    },
    {
      "type": "translation",
      "id": 1,
      "src": "/path/to/srclang1-tgtlang1/train.srclang1",
      "tgt": "/path/to/srclang1-tgtlang1/train.tgtlang1"
    },
    {
      "type": "translation",
      "id": 0,
      "src": "/path/to/srclang2-tgtlang0/train.srclang2",
      "tgt": "/path/to/srclang2-tgtlang0/train.tgtlang0"
    },
    {
      "type": "translation",
      "id": 1,
      "src": "/path/to/srclang2-tgtlang1/train.srclang2",
      "tgt": "/path/to/srclang2-tgtlang1/train.tgtlang1"
    },
    ...
  ],
  "valid": [
    {
      "type": "translation",
      "id": 0,
      "src": "/unused",
      "tgt": "/unused"
    }
  ]
}
```
where paths are paths to binarized indexed fairseq dataset files.
`id` represents the target language id.


## Training Command Line Example

```
fairseq-train \
  /path/to/configfile_described_above.json \
  --user-dir examples/laser/laser_src \
  --log-interval 100 --log-format simple \
  --task laser --arch laser_lstm \
  --save-dir . \
  --optimizer adam \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --clip-norm 5 \
  --warmup-updates 90000 \
  --update-freq 2 \
  --dropout 0.0 \
  --encoder-dropout-out 0.1 \
  --max-tokens 2000 \
  --max-epoch 50 \
  --encoder-bidirectional \
  --encoder-layers 5 \
  --encoder-hidden-size 512 \
  --decoder-layers 1 \
  --decoder-hidden-size 2048 \
  --encoder-embed-dim 320 \
  --decoder-embed-dim 320 \
  --decoder-lang-embed-dim 32 \
  --warmup-init-lr 0.001 \
  --disable-validation
```


## Applications

We showcase several applications of multilingual sentence embeddings
with code to reproduce our results (in the directory "tasks").

* [**Cross-lingual document classification**](https://github.com/facebookresearch/LASER/tree/master/tasks/mldoc) using the
  [*MLDoc*](https://github.com/facebookresearch/MLDoc) corpus [2,6]
* [**WikiMatrix**](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix)
   Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia [7]
* [**Bitext mining**](https://github.com/facebookresearch/LASER/tree/master/tasks/bucc) using the
  [*BUCC*](https://comparable.limsi.fr/bucc2018/bucc2018-task.html) corpus [3,5]
* [**Cross-lingual NLI**](https://github.com/facebookresearch/LASER/tree/master/tasks/xnli)
  using the [*XNLI*](https://www.nyu.edu/projects/bowman/xnli/) corpus [4,5,6]
* [**Multilingual similarity search**](https://github.com/facebookresearch/LASER/tree/master/tasks/similarity) [1,6]
* [**Sentence embedding of text files**](https://github.com/facebookresearch/LASER/tree/master/tasks/embed)
  example how to calculate sentence embeddings for arbitrary text files in any of the supported language.

**For all tasks, we use exactly the same multilingual encoder, without any task specific optimization or fine-tuning.**



## References

[1] Holger Schwenk and Matthijs Douze,
    [*Learning Joint Multilingual Sentence Representations with Neural Machine Translation*](https://aclanthology.info/papers/W17-2619/w17-2619),
    ACL workshop on Representation Learning for NLP, 2017

[2] Holger Schwenk and Xian Li,
    [*A Corpus for Multilingual Document Classification in Eight Languages*](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf),
    LREC, pages 3548-3551, 2018.

[3] Holger Schwenk,
    [*Filtering and Mining Parallel Data in a Joint Multilingual Space*](http://aclweb.org/anthology/P18-2037)
    ACL, July 2018

[4] Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk and Veselin Stoyanov,
    [*XNLI: Cross-lingual Sentence Understanding through Inference*](https://aclweb.org/anthology/D18-1269),
    EMNLP, 2018.

[5] Mikel Artetxe and Holger Schwenk,
    [*Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*](https://arxiv.org/abs/1811.01136)
    arXiv, Nov 3 2018.

[6] Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464)
    arXiv, Dec 26 2018.

[7] Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong and Paco Guzman,
    [*WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia*](https://arxiv.org/abs/1907.05791)
    arXiv, July 11  2019.

[8] Holger Schwenk, Guillaume Wenzek, Sergey Edunov, Edouard Grave and Armand Joulin
    [*CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB*](https://arxiv.org/abs/1911.04944)
