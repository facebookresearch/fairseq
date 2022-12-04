#!/bin/bash

set -x

# /fsx-wav2vec/wnhsu/d2v2/base_roberta/bpe/1M/base_8/dataset.batch_size:16/optimization.lr:0.0003/optimization.max_update:1000000
# COLA            QNLI            MRPC            RTE             SST_2
# ('8e-6', 82.3)  ('2e-5', 89.7)  ('2e-5', 86.3)  ('8e-6', 72.6)  ('8e-6', 92.3)  84.64   87.65

root=/fsx-wav2vec/wnhsu/d2v2/base_roberta/bpe/1M/base_8/dataset.batch_size:16/optimization.lr:0.0003/optimization.max_update:1000000

# # CoLA: 8e-6
# eval_path=${root}/finetune_lr/cola/8e-6/0/checkpoints/checkpoint_best.pt
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py \
#   --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp +run_config=local \
#   common_eval.path=${eval_path} task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin dataset.batch_size=64 \
#   dataset.required_batch_size_multiple=1

# # MRPC: 2e-5
# eval_path=${root}/finetune_lr/mrpc/2e-5/0/checkpoints/checkpoint_best.pt
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py \
#   --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp +run_config=local \
#   common_eval.path=${eval_path} task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/MRPC-bin dataset.batch_size=64 \
#   dataset.required_batch_size_multiple=1


# /fsx-wav2vec/wnhsu/d2v2/base_multi/bpe/1M_swp2_bs-and-mask/base_text_only_task_pgrp_1M_8/dataset.batch_size:4/model.clone_batch:8/model.modalities.text.mask_length:1/model.modalities.text.mask_prob:0.48
# COLA            QNLI            MRPC            RTE             SST_2
# ('2e-5', 84.2)  ('2e-5', 90.8)  ('2e-5', 87.0)  ('1e-5', 67.9)  ('2e-5', 92.8)  84.54   88.70

# root=/fsx-wav2vec/wnhsu/d2v2/base_multi/bpe/1M_swp2_bs-and-mask/base_text_only_task_pgrp_1M_8/dataset.batch_size:4/model.clone_batch:8/model.modalities.text.mask_length:1/model.modalities.text.mask_prob:0.48

# # CoLA: 2e-5
# eval_path=${root}/finetune_lr/cola/2e-5/0/checkpoints/checkpoint_best.pt
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py \
#   --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp +run_config=local \
#   common_eval.path=${eval_path} task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin dataset.batch_size=64 \
#   dataset.required_batch_size_multiple=1

# MRPC: 2e-5
# eval_path=${root}/finetune_lr/mrpc/2e-5/0/checkpoints/checkpoint_best.pt
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py \
#   --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp +run_config=local \
#   common_eval.path=${eval_path} task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/MRPC-bin dataset.batch_size=64 \
#   dataset.required_batch_size_multiple=1


#### Example for computing MNLI non-matched acc
# valid = matched
# valid1 = mismatched
PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py \
  --config-dir examples/data2vec/config/multi --config-name base_text_only_task +run_config=local \
  common_eval.path=/fsx-wav2vec/wnhsu/d2v2/base_multi_speed/v1_default/base_text_only_task_8/finetune_sweep2/mnli/2e-5/run_1/0/checkpoints/checkpoint_best.pt \
  task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/MNLI-bin dataset.batch_size=64  dataset.required_batch_size_multiple=1 \
  dataset.valid_subset=valid1 
