# (Vectorized) Lexically constrained decoding with dynamic beam allocation

This page provides instructions for how to use lexically constrained decoding in Fairseq.
Fairseq implements the code described in the following papers:

* [Fast Lexically Constrained Decoding With Dynamic Beam Allocation](https://www.aclweb.org/anthology/N18-1119/) (Post & Vilar, 2018)
* [Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting](https://www.aclweb.org/anthology/N19-1090/) (Hu et al., 2019)

## Quick start

Constrained search is enabled by adding the command-line argument `--constraints` to `fairseq-interactive`.
Constraints are appended to each line of input, separated by tabs. Each constraint (one or more tokens)
is a separate field.

The following command, using [Fairseq's WMT19 German--English model](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md),
translates the sentence *Die maschinelle Übersetzung ist schwer zu kontrollieren.* with the constraints
"hard" and "to influence".

    echo -e "Die maschinelle Übersetzung ist schwer zu kontrollieren.\thard\ttoinfluence" \
    | normalize.py | tok.py \
    | fairseq-interactive /path/to/model \
      --path /path/to/model/model1.pt \
      --bpe fastbpe \
      --bpe-codes /path/to/model/bpecodes \
      --constraints \
      -s de -t en \
      --beam 10

(tok.py and normalize.py can be found in the same directory as this README; they are just shortcuts around Fairseq's WMT19 preprocessing).
This will generate the following output:

    [snip]
    S-0     Die masch@@ in@@ elle Über@@ setzung ist schwer zu kontrollieren .
    W-0     1.844   seconds
    C-0     hard
    C-0     influence
    H-0     -1.5333266258239746     Mach@@ ine trans@@ lation is hard to influence .
    D-0     -1.5333266258239746     Machine translation is hard to influence .
    P-0     -0.5434 -0.1423 -0.1930 -0.1415 -0.2346 -1.8031 -0.1701 -11.7727 -0.1815 -0.1511

By default, constraints are generated in the order supplied, with any number (zero or more) of tokens generated
between constraints. If you wish for the decoder to order the constraints, then use `--constraints unordered`.
Note that you may want to use a larger beam.

## Implementation details

The heart of the implementation is in `fairseq/search.py`, which adds a `LexicallyConstrainedBeamSearch` instance.
This instance of beam search tracks the progress of each hypothesis in the beam through the set of constraints
provided for each input sentence. It does this using one of two classes, both found in `fairseq/token_generation_contstraints.py`:

* OrderedConstraintState: assumes the `C` input constraints will be generated in the provided order
* UnorderedConstraintState: tries to apply `C` (phrasal) constraints in all `C!` orders

## Differences from Sockeye

There are a number of [differences from Sockeye's implementation](https://awslabs.github.io/sockeye/inference.html#lexical-constraints).

* Generating constraints in the order supplied (the default option here) is not available in Sockeye.
* Due to an improved beam allocation method, there is no need to prune the beam.
* Again due to better allocation, beam sizes as low as 10 or even 5 are often sufficient.
* [The vector extensions described in Hu et al.](https://github.com/edwardjhu/sockeye/tree/trie_constraints) (NAACL 2019) were never merged
  into the main Sockeye branch.

## Citation

The paper first describing lexical constraints for seq2seq decoding is:

```bibtex
@inproceedings{hokamp-liu-2017-lexically,
  title = "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search",
  author = "Hokamp, Chris  and
    Liu, Qun",
  booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2017",
  address = "Vancouver, Canada",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/P17-1141",
  doi = "10.18653/v1/P17-1141",
  pages = "1535--1546",
}
```

The fairseq implementation uses the extensions described in

```bibtex
@inproceedings{post-vilar-2018-fast,
    title = "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation",
    author = "Post, Matt  and
      Vilar, David",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N18-1119",
    doi = "10.18653/v1/N18-1119",
    pages = "1314--1324",
}
```

and

```bibtex
@inproceedings{hu-etal-2019-improved,
  title = "Improved Lexically Constrained Decoding for Translation and Monolingual Rewriting",
  author = "Hu, J. Edward  and
    Khayrallah, Huda  and
    Culkin, Ryan  and
    Xia, Patrick  and
    Chen, Tongfei  and
    Post, Matt  and
    Van Durme, Benjamin",
  booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
  month = jun,
  year = "2019",
  address = "Minneapolis, Minnesota",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/N19-1090",
  doi = "10.18653/v1/N19-1090",
  pages = "839--850",
}
```
