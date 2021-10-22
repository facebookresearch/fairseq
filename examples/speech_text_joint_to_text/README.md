# Joint Speech Text training in Fairseq
An extension of Fairseq s2t project with the speech to text task enhanced by the co-trained text to text mapping task. More details about Fairseq s2t can be found [here](../speech_to_text/README.md)

## Examples
Examples of speech text joint training in fairseq
- [English-to-German MuST-C model](docs/ende-mustc.md)
- [IWSLT 2021 Multilingual Speech Translation](docs/iwslt2021.md)

## Citation
Please cite as:
```
@inproceedings{Tang2021AGM,
  title={A General Multi-Task Learning Framework to Leverage Text Data for Speech to Text Tasks},
  author={Yun Tang and J. Pino and Changhan Wang and Xutai Ma and Dmitriy Genzel},
  booktitle={ICASSP},
  year={2021}
}

@inproceedings{Tang2021IST,
  title = {Improving Speech Translation by Understanding and Learning from the Auxiliary Text Translation Task},
  author = {Yun Tang and Juan Pino and Xian Li and Changhan Wang and Dmitriy Genzel},
  booktitle = {ACL},
  year = {2021},
}

@inproceedings{Tang2021FST,
  title = {FST: the FAIR Speech Translation System for the IWSLT21 Multilingual Shared Task},
  author = {Yun Tang and Hongyu Gong and Xian Li and Changhan Wang  and Juan Pino and  Holger Schwenk and  Naman Goyal},
  booktitle = {IWSLT},
  year = {2021},
}

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
