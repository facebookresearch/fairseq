 python fairseq_cli/hydra_train.py \
  --config-dir ./examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint