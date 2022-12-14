# Multilingual pretraining RoBERTa

This tutorial will walk you through pretraining multilingual RoBERTa.

### 1) Preprocess the data

```bash
DICTIONARY="/private/home/namangoyal/dataset/XLM/wiki/17/175k/vocab"
DATA_LOCATION="/private/home/namangoyal/dataset/XLM/wiki/17/175k"

for LANG in en es it
do
  fairseq-preprocess \
      --only-source \
      --srcdict $DICTIONARY \
      --trainpref "$DATA_LOCATION/train.$LANG" \
      --validpref "$DATA_LOCATION/valid.$LANG" \
      --testpref "$DATA_LOCATION/test.$LANG" \
      --destdir "wiki_17-bin/$LANG" \
      --workers 60;
done
```

### 2) Train RoBERTa base

[COMING UP...]
