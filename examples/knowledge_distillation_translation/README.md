## Knowledge Distillation Translation

This README contains instructions for distilling a seq2seq model (especially transformers)

Here is a bash script that can distill the knowledge of a large transformer model to a smaller transformer-base model.

The student transformer being trained is compatible with factorized embeddings and should share the vocabularies with the teacher.

The KD implementation has been sourced from [here](https://github.com/LeslieOverfitting/selective_distillation) and is based on the following: [paper](https://arxiv.org/abs/2105.12967)

```
fairseq-train $bin_dir \
--max-source-positions 210 --max-target-positions 210 --max-update 1000000 --max-tokens 8192 \
--arch transformer --activation-fn gelu --dropout 0.2 \
--task kd_translation --distil-strategy word_and_seq_level --teacher-temp 2 --student-temp 2 --teacher-checkpoint-path $teacher_checkpoint_dir \
--criterion kd_label_smoothed_cross_entropy --label-smoothing 0.1 --alpha 0.5 \
--source-lang SRC --target-lang TGT \
--lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 --warmup-init-lr 1e-07 --lr 0.0005 --warmup-updates 4000 \
--save-dir $student_save_dir --save-interval 1 --keep-last-epochs 1 --patience 5 \
--skip-invalid-size-inputs-valid-test \
--fp16 \
--update-freq 8 \
--distributed-world-size 1 \
```