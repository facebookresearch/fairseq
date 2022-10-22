XStoryCloze consists of the professionally translated version of the [English StoryCloze dataset](https://cs.rochester.edu/nlp/rocstories/) (Spring 2016 version) to 10 non-English languages. This dataset is released by Meta AI.

# Languages
ru, zh (Simplified), es (Latin America), ar, hi, id, te, sw, eu, my.

# Data Splits
This dataset is intended to be used for evaluating the zero- and few-shot learning capabilities of multlingual language models. We split the data for each language into train and test (360 vs. 1510 examples, respectively). The released data files for different languages maintain a line-by-line alignment.

# Access English StoryCloze
Please request the original English StoryCloze dataset through the [official channel](https://cs.rochester.edu/nlp/rocstories/). You can create a split of the en data following our data split scheme using the following commands:
```
head -361 spring2016.val.tsv > spring2016.val.en.tsv.split_20_80_train.tsv

head -1 spring2016.val.tsv > spring2016.val.en.tsv.split_20_80_eval.tsv
tail -1510 spring2016.val.tsv >> spring2016.val.en.tsv.split_20_80_eval.tsv
```

# Licence
XStoryCloze is opensourced under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode), the same license as the original English StoryCloze.

# Citation
If you use XStoryCloze in your work, please cite
```
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
