# The examples below show how a feedforward/SNAIL model can be adapted to new 
# tasks by either no-finetuning, full-finetuning or fine-tuning only the task
# embedding (z)

# 1) Full-finetuning a multi-task trained model
# LOAD=ff_multi
# MODE=single_task
# MDL=default
# FINETUNE=yes
# TEST_SAMPLES=4
# 
# 2) Fine-tune only z in a meta-trained model
# LOAD=ff_meta
# MODE=single_task
# MDL=default
# FINETUNE=z
# TEST_SAMPLES=4
# 
# 3) Evaluate a multi-task trained SNAIL model without any fine-tuning
# LOAD=snail_multi_ex4
# MODE=single_task
# MDL=snail
# FINETUNE=no
# TEST_SAMPLES=4
# 
# 4) Meta trained SNAIL + fine-tuning only z
# LOAD=snail_meta_ex4
# MODE=single_task
# MDL=snail
# FINETUNE=z
# TEST_SAMPLES=4
# 
# To full-finetune a multi-task trained model, use 1)
# Eg: 
# 
LOAD=ff_multi
MODE=single_task
MDL=default
FINETUNE=yes
TEST_SAMPLES=4

## DEFALUTS ##
CKPT_DIR=/checkpoint/llajan/final_models
DBG_MODE=no
LAYERS=4
ZSIZE=128

export CKPT_DIR=$CKPT_DIR
export LOAD=$LOAD
export MODE=$MODE
export MDL=$MDL
export FINETUNE=$FINETUNE
export TEST_SAMPLES=$TEST_SAMPLES
export DBG_MODE=$DBG_MODE
export LAYERS=$LAYERS
export ZSIZE=$ZSIZE
export CUDA_VISIBLE_DEVICES=0

./scripts/eval.sh
