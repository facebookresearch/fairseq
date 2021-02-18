# Speech-to-Text (S2T) Modeling

[https://www.aclweb.org/anthology/2020.aacl-demo.6](https://www.aclweb.org/anthology/2020.aacl-demo.6.pdf)

Speech recognition (ASR) and speech-to-text translation (ST) with fairseq.

## Data Preparation
S2T modeling data consists of source speech features, target text and other optional information
(source text, speaker id, etc.). Fairseq S2T uses per-dataset-split TSV manifest files
to store these information. Each data field is represented by a column in the TSV file.

Unlike text token embeddings, speech features (e.g. log mel-scale filter banks) are usually fixed
during model training and can be pre-computed. The manifest file contains the path to
either the feature file in NumPy format or the WAV/FLAC audio file. For the latter,
features will be extracted on-the-fly by fairseq S2T. Optionally, feature/audio files can be packed
into uncompressed ZIP files (then accessed via byte offset and length) to improve I/O performance.

Fairseq S2T also employs a YAML file for data related configurations: tokenizer type and dictionary path
for the target text, feature transforms such as CMVN (cepstral mean and variance normalization) and SpecAugment,
temperature-based resampling, etc.

## Model Training
Fairseq S2T uses the unified `fairseq-train` interface for model training. It requires arguments `--task speech_to_text`,
 `--arch <model architecture in fairseq.models.speech_to_text.*>` and `--config-yaml <config YAML filename>`.

## Inference & Evaluation
Fairseq S2T uses the unified `fairseq-generate`/`fairseq-interactive` interface for inference and evaluation. It
requires arguments `--task speech_to_text` and `--config-yaml <config YAML filename>`. The interactive console takes
audio paths (one per line) as inputs.


## Examples
- [Speech Recognition (ASR) on LibriSpeech](docs/librispeech_example.md)

- [Speech-to-Text Translation (ST) on MuST-C](docs/mustc_example.md)

- [Speech-to-Text Translation (ST) on CoVoST 2](docs/covost_example.md)

- [Speech-to-Text Translation (ST) on Multilingual TEDx](docs/mtedx_example.md)

## Updates
- 02/04/2021: Added interactive decoding (`fairseq-interactive`) support. Examples:
  [ASR (LibriSpeech)](docs/librispeech_example.md#interactive-decoding)
  and [ST (CoVoST 2)](docs/covost_example.md#interactive-decoding).
- 01/08/2021: Several fixes for S2T Transformer model, inference-time de-tokenization, scorer configuration and data
  preparation scripts. We also add pre-trained models to the examples and revise the instructions.
  Breaking changes: the data preparation scripts now extract filterbank features without CMVN. CMVN is instead applied
  on-the-fly (defined in the config YAML).

## What's Next
- We are migrating the old fairseq [ASR example](../speech_recognition) into this S2T framework and
  merging the features from both sides.
- The following papers also base their experiments on fairseq S2T. We are adding more examples for replication.
  - [Improving Cross-Lingual Transfer Learning for End-to-End Speech Recognition with Speech Translation (Wang et al., 2020)](https://arxiv.org/abs/2006.05474)
  - [Self-Supervised Representations Improve End-to-End Speech Translation (Wu et al., 2020)](https://arxiv.org/abs/2006.12124)
  - [Self-Training for End-to-End Speech Translation (Pino et al., 2020)](https://arxiv.org/abs/2006.02490)
  - [CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus (Wang et al., 2020)](https://arxiv.org/abs/2002.01320)
  - [Harnessing Indirect Training Data for End-to-End Automatic Speech Translation: Tricks of the Trade (Pino et al., 2019)](https://arxiv.org/abs/1909.06515)

## Citation
Please cite as:
```
@inproceedings{wang2020fairseqs2t,
  title = {fairseq S2T: Fast Speech-to-Text Modeling with fairseq},
  author = {Changhan Wang and Yun Tang and Xutai Ma and Anne Wu and Dmytro Okhonko and Juan Pino},
  booktitle = {Proceedings of the 2020 Conference of the Asian Chapter of the Association for Computational Linguistics (AACL): System Demonstrations},
  year = {2020},
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
