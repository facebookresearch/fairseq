#!/usr/bin/env bash
rm -rf fsdp_dummy
mkdir -p fsdp_dummy
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train /private/home/sshleifer/data-bin/stories_mmap \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10 \
    --max-update 5 --log-format json --log-interval 1 \
    --save-interval-updates 5 --save-dir fsdp_dummy --disable-validation \
    --restore-file x.pt "$@"

# Now we try to load the checkpoint
CUDA_VISIBLE_DEVICES=0,1 fairseq-train /private/home/sshleifer/data-bin/stories_mmap \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 256 --batch-size 8 \
    --arch transformer_lm_gpt2_tiny \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10 \
    --max-update 2 --log-format json --log-interval 1 \
    --save-interval-updates 2 --save-dir fsdp_dummy
