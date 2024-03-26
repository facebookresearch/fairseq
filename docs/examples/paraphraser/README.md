# Paraphrasing with round-trip translation and mixture of experts

Machine translation models can be used to paraphrase text by translating it to
an intermediate language and back (round-trip translation).

This example shows how to paraphrase text by first passing it to an
English-French translation model, followed by a French-English [mixture of
experts translation model](/examples/translation_moe).

##### 0. Setup

Clone fairseq from source and install necessary dependencies:
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --editable .
pip install sacremoses sentencepiece
```

##### 1. Download models
```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/paraphraser.en-fr.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/paraphraser.fr-en.hMoEup.tar.gz
tar -xzvf paraphraser.en-fr.tar.gz
tar -xzvf paraphraser.fr-en.hMoEup.tar.gz
```

##### 2. Paraphrase
```bash
python examples/paraphraser/paraphrase.py \
    --en2fr paraphraser.en-fr \
    --fr2en paraphraser.fr-en.hMoEup
# Example input:
#   The new date for the Games, postponed for a year in response to the coronavirus pandemic, gives athletes time to recalibrate their training schedules.
# Example outputs:
#   Delayed one year in response to the coronavirus pandemic, the new date of the Games gives athletes time to rebalance their training schedule.
#   The new date of the Games, which was rescheduled one year in response to the coronavirus (CV) pandemic, gives athletes time to rebalance their training schedule.
#   The new date of the Games, postponed one year in response to the coronavirus pandemic, provides athletes with time to rebalance their training schedule.
#   The Games' new date, postponed one year in response to the coronavirus pandemic, gives athletes time to rebalance their training schedule.
#   The new Games date, postponed one year in response to the coronavirus pandemic, gives the athletes time to rebalance their training schedule.
#   The new date of the Games, which was postponed one year in response to the coronavirus pandemic, gives the athletes time to rebalance their training schedule.
#   The new date of the Games, postponed one year in response to the coronavirus pandemic, gives athletes time to rebalance their training schedule.
#   The new date of the Games, postponed one year in response to the coronavirus pandemic, gives athletes time to re-balance their training schedule.
#   The new date of the Games, postponed one year in response to the coronavirus pandemic, gives the athletes time to rebalance their schedule of training.
#   The new date of the Games, postponed one year in response to the pandemic of coronavirus, gives the athletes time to rebalance their training schedule.
```
