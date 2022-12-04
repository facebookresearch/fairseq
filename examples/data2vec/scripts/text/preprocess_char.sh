#!/bin/bash

set -eu

# ===== BPE to character data (pretrain and finetune)
python examples/data2vec/scripts/text/unprocess_data.py


# ===== binarize pretraining data
raw_root="/checkpoint/wnhsu/data/data2vec2/data/text/bookwiki_aml-full-mmap2-txt"
bin_root="/checkpoint/wnhsu/data/data2vec2/data/text/bookwiki_aml-full-mmap2-bin"

# "this is a book" -> "t h i s | i s | a | b o o k"
sed -e 's# #|#g' -e 's#\(.\)#\1 #g' -i ${raw_root}/train*
sed -e 's# #|#g' -e 's#\(.\)#\1 #g' -i ${raw_root}/valid*
cat ${raw_root}/train{0,1,2,3,4}.txt > ${raw_root}/train_all.txt

# process
PYTHONPATH=$(pwd) python fairseq_cli/preprocess.py \
  --dataset-impl mmap --trainpref ${raw_root}/train_all.txt --validpref ${raw_root}/valid.txt \
  --workers 70 --only-source --destdir ${bin_root}


# ===== binarize finetuning data
src_root="/fsx-wav2vec/wnhsu/data/data2vec2/data/text/GLUE"
raw_root="/fsx-wav2vec/wnhsu/data/data2vec2/data/text/GLUE_chr"
src_dict="/fsx-wav2vec/wnhsu/data/data2vec2/data/text/bookwiki_aml-full-mmap2-chr-bin/dict.txt"

# "this is a book" -> "t h i s | i s | a | b o o k"
sed -e 's# #|#g' -e 's#\(.\)#\1 #g' -i ${raw_root}/*/*/{train,valid,test}*

# process
for d in ${raw_root}/*/input*; do
  echo $d
  PYTHONPATH=$(pwd) python fairseq_cli/preprocess.py \
    --dataset-impl mmap --srcdict $src_dict \
    --trainpref $d/train.txt --validpref $d/valid.txt --testpref $d/test.txt \
    --workers 10 --only-source --destdir $d
done

# symbolic links for labels
for name in CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B; do
  ln -s ${src_root}/${name}-bin/label ${raw_root}/${name}-bin/
done
