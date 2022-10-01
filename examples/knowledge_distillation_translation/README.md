## Knowledge Distillation Translation

This README contains instructions for distilling a seq2seq model (especially transformers)

Here is a bash script that can distill the knowledge of a large transformer model to a smaller transformer-base model.

The student transformer being trained is compatible with factorized embeddings and should share the vocabularies with the teacher.

The KD implementation has been sourced from [here](https://github.com/LeslieOverfitting/selective_distillation) and is based on the following: [paper](https://arxiv.org/abs/2105.12967)

```
fairseq-train $data_dir \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--arch transformer --activation-fn gelu --use-factorized-embedding --factorized-embedding-dim 128 --layernorm-factorized-embedding \
--task kd_translation --distillation-strategy batch_level --distillation-rate 0.5 --temperature 2 --temperature-schedule none --teacher-checkpoint-path $teacher_checkpoint_path \
--criterion kd_label_smoothed_cross_entropy --label-smoothing 0.1 --alpha 10 \
--source-lang SRC --target-lang TGT \
--lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 --warmup-init-lr 1e-07 --lr 0.0005 --warmup-updates 4000 \
--dropout 0.2 \
--save-dir $student_save_dir \
--save-interval 1 \
--keep-last-epochs 5 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--memory-efficient-fp16 \
--update-freq 1 \
--distributed-world-size 1 \
--max-tokens 2048 \
--user-dir $custom_user_dir
```