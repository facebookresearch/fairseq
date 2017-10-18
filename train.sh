OUT=/home/playma/4t/playma/Experiment/fairseq-py/data-bin
MODEL=/home/playma/4t/playma/Experiment/fairseq-py/checkpoints/20171016_2230

CUDA_VISIBLE_DEVICES=0 python3 train.py $OUT \
  --lr 0.5 --clip-norm 0.1 --dropout 0 --max-tokens 4000 \
  --arch fconv_iwslt_de_en --save-dir $MODEL \
  --max-epoch 100 \
  > $MODEL/log.txt

