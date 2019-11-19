#!/bin/bash


#training_data="/raid/data/daga01/fairseq_train/data-bin-32k-red-lazy"
training_data="/raid/data/daga01/fairseq_train/data-bin-50k-red-lazy_7"

checkpoints="/raid/data/daga01/fairseq_train/checkpoints/fconv"



#call_train_distributed() {
##For example, to train a large English-German Transformer model on 2 nodes each with 8 GPUs (in total 16 GPUs), run the following command on each node, replacing node_rank=0 with node_rank=1 on the second node:
## orig: --nnodes=2 --nproc_per_node=8
#
#python -m torch.distributed.launch --nproc_per_node=4 \
#    --nnodes=1 --node_rank=0 --master_addr="192.168.1.1" \
#    --master_port=1234 \
#    CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
#    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#    --lr 0.0005 --min-lr 1e-09 \
#    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 3584  --save-dir ${checkpoints}
#}


call_simplest(){
CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-train "$training_data" \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir "$checkpoints"
}


call_train_big(){
# 8 GPU cumul/update-freq 16 ?= 4 GPU update-freq 32 
# lr=5e-4 - orig; here I'll change to 2xlr=1e-3 as stated in the paper

CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.001 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584  --save-dir ${checkpoints} --keep-last-epochs 3  \
    --dataset-impl lazy 
    #--ddp-backend no_c10d --update-freq 32 
}


call_train_big_wide(){
# lr=5e-4 - orig; here I don't change to 2xlr=1e-3 as stated in the 2018 paper because my batches are not so big

CUDA_VISIBLE_DEVICES=1,2,3,4  $(which fairseq-train) ${training_data} \
    --arch transformer_vaswani_wmt_en_de_big_wide --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584  --save-dir ${checkpoints} --keep-last-epochs 10  \
    --dataset-impl lazy --max-epoch 45
    #--update-freq 32 --max-epoch 3
    #--num-workers 4 
    #--update-freq 8 
}


echo $(which python)

LOG0="/raid/data/daga01/fairseq_train/LOG_test"
time -p call_train_big > $LOG0 2>&1

#LOG="/raid/data/daga01/fairseq_train/LOG_R2_32k_wide"
#time -p call_train_big_wide > $LOG 2>&1
