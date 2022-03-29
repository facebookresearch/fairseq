Speech Synthesis (S^2)
===
[https://arxiv.org/abs/2109.06912](https://arxiv.org/abs/2109.06912)

Speech synthesis with fairseq.

## Features

- Autoregressive and non-autoregressive models
- Multi-speaker synthesis
- Audio preprocessing (denoising, VAD, etc.) for less curated data
- Automatic metrics for model development
- Similar data configuration as [S2T](../speech_to_text/README.md)


## Examples
- [Single-speaker synthesis on LJSpeech](docs/ljspeech_example.md)
- [Multi-speaker synthesis on VCTK](docs/vctk_example.md)
- [Multi-speaker synthesis on Common Voice](docs/common_voice_example.md)


## Citation
Please cite as:
```
@article{wang2021fairseqs2,
  title={fairseq S\^{} 2: A Scalable and Integrable Speech Synthesis Toolkit},
  author={Wang, Changhan and Hsu, Wei-Ning and Adi, Yossi and Polyak, Adam and Lee, Ann and Chen, Peng-Jen and Gu, Jiatao and Pino, Juan},
  journal={arXiv preprint arXiv:2109.06912},
  year={2021}
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
