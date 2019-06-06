# Training with different models and training algorithms
# The 4 examples below show how a feedforward/SNAIL model can be trained with
# multitask training or meta training algorithms

# # 1) Feedforward model, multitask training
# EXP_NAME=ff_multi
# MODE=multitask
# MDL=default
# 
# # 2) Feedforward model, meta training
# EXP_NAME=ff_meta
# MODE=meta_zeroinit
# MDL=default
# 
# # 3) SNAIL model, multitask training
# # Train with at most 4 training context sequences
# EXP_NAME=snail_multi_ex4
# MODE=multitask
# MDL=snail
# TEST_SAMPLES=4
# 
# # 4) SNAIL model, meta training
# EXP_NAME=snail_meta_ex4
# MODE=meta
# MDL=snail
# TEST_SAMPLES=4

# To train a feedforward model with multitask training, use 1)
# Eg: 

EXP_NAME=ff_multi
MODE=multitask
MDL=default

## DEFALUTS ##
CKPT_DIR=/checkpoint/llajan/final_models
DBG_MODE=no
LAYERS=4
ZSIZE=128
NUMGRADS=25

export CKPT_DIR=$CKPT_DIR
export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL
export TEST_SAMPLES=$TEST_SAMPLES
export DBG_MODE=$DBG_MODE
export LAYERS=$LAYERS
export ZSIZE=$ZSIZE
export NUMGRADS=$NUMGRADS
export CUDA_VISIBLE_DEVICES=0

./scripts/train.sh
