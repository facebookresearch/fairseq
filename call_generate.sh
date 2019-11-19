#!/bin/bash


top="/raid/data/daga01/fairseq_train"

#model_time="fconv_R1"
model_time="snapshot"

#model="checkpoint_best.pt"
model="checkpoint38.pt"

#direction="data-bin-32k-red-lazy"
direction="data-bin-32k-red-lazy-copy"


generate_big_sacrebleu(){
CUDA_VISIBLE_DEVICES=5,6 fairseq-generate $top/${direction} \
    --path $top/checkpoints/${model_time}/${model} \
    --batch-size 128 --beam 4 --lenpen 0.6  --sacrebleu
}


generate_big(){
CUDA_VISIBLE_DEVICES=5,6 fairseq-generate $top/${direction} \
    --path $top/checkpoints/${model_time}/${model} \
    --batch-size 128 --beam 4 --lenpen 0.6   
}



generate_big_5(){
CUDA_VISIBLE_DEVICES=5,6 fairseq-generate $top/${direction} \
    --path $top/checkpoints/${model_time}/${model} \
    --batch-size 128 --beam 5
}


generate_big_5_sacrebleu(){
CUDA_VISIBLE_DEVICES=5,6 fairseq-generate $top/${direction} \
    --path $top/checkpoints/${model_time}/${model} \
    --batch-size 128 --beam 5 --sacrebleu
}





OUT="/raid/data/daga01/fairseq_train/OUT_sacrebleu"
time -p generate_big_sacrebleu > $OUT 2>&1

OUT2="/raid/data/daga01/fairseq_train/OUT"
time -p generate_big > $OUT2 2>&1

OUT3="/raid/data/daga01/fairseq_train/OUT_5_sacrebleu"
time -p generate_big_5_sacrebleu > $OUT3 2>&1

OUT4="/raid/data/daga01/fairseq_train/OUT_5"
time -p generate_big_5 > $OUT4 2>&1
