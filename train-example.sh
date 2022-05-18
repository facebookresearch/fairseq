dp=2
gas=1
max_tokens=1024
zero=3

save_dir=z3-50k

#(max_tokens / tokens_per_sample) * world_size * update_freq = global_batch_size

#  --save-interval-updates 10 \
fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --distributed-world-size $dp \
  --seed 100 \
  --tensorboard-logdir ${save_dir}/tsb/dp${dp}_gas${gas}_mt${max_tokens}_z${zero}_50k \
  --save-dir ${save_dir}/transformer_wikitext-103_dp${dp}_gas${gas}_mt${max_tokens}_z${zero}_50k \
  --arch transformer_lm \
  --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 \
  --sample-break-mode none \
  --max-tokens ${max_tokens} \
  --update-freq ${gas} \
  --fp16 \
  --deepspeed \
  --zero ${zero} \
  --max-update 50000
