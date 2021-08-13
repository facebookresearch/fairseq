# Training MoE language models

## Dependencies

Follow the fairseq installation instructions:
https://github.com/pytorch/fairseq/#requirements-and-installation

The following package versions are recommended:

apex:
```bash
pip install -v --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" --global-option="--fast_multihead_attn" \
    git+git://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690
```

fairscale:
```bash
pip install fairscale==0.4.0
```

hydra:
```bash
pip install hydra-core==1.0.7 omegaconf==2.0.6
```

megatron (must be installed from source to get fused kernels):
```bash
git clone --depth=1 --branch v2.4 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

## Single-node training

The following command will benchmark an MoE language model using synthetic data
on 8 GPUs. The model has 8 experts (one per GPU) and 4.1B parameters total.

```bash
# set NUM_EXPERTS based on # of GPUs and desired # experts per GPU
# generally it's recommended to have a single expert per GPU
NUM_EXPERTS=8
TOKENS_PER_SAMPLE=2048
python fairseq_cli/train.py \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
  --task dummy_lm --tokens-per-sample $TOKENS_PER_SAMPLE \
  --arch transformer_lm_gpt --share-decoder-input-output-embed \
  --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 \
  --decoder-attention-heads 32 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
  --optimizer adam --fp16-adam-stats --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --warmup-updates 750 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 4 --update-freq 1 \
  --max-update 250 --disable-validation \
  --log-format json --log-interval 10
```

The total parameter count can be infered from the logs:
```
(...)
2021-08-13 14:54:20 | INFO | fairseq_cli.train | num. non-expert model params: 908,423,168 (num. trained: 908,423,168)
2021-08-13 14:54:20 | INFO | fairseq_cli.train | num. expert model params: 402,776,064 (num. trained: 402,776,064)
(...)
```
The expert params are distinct on each GPU, so the total parameter count is `908M + 8 * 403M = 4.1B`

**Sample output on 8 x V100:**
```
2021-08-13 14:58:39 | INFO | fairseq.modules.fused_bias_gelu | Done with compiling and loading fused kernels.
2021-08-13 14:58:44 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 64.0
2021-08-13 14:58:49 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 32.0
2021-08-13 14:58:53 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 16.0
2021-08-13 14:59:32 | INFO | train_inner | {"epoch": 1, "update": 0.004, "loss": "20.714", "moe_gate_loss": "16.7217", "overflow_expert1": "20.84", "overflow_expert2": "53.493", "entropy_gating": "1.943", "expert1_balance_top": "66.521", "expert1_balance_bottom": "2.528", "unused_expert1_count": "0.12", "expert2_balance_top": "50.142", "expert2_balance_bottom": "5.417", "unused_expert2_count": "0.052", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "20.472", "ppl": "1.45489e+06", "wps": "16606.5", "ups": "0.25", "wpb": "65536", "bsz": "32", "num_updates": "10", "lr": "7.33333e-06", "gnorm": "30.01", "loss_scale": "16", "train_wall": "62", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "19", "cuda_gb_free": "21.4", "wall": "68"}
2021-08-13 15:00:12 | INFO | train_inner | {"epoch": 1, "update": 0.007, "loss": "16.194", "moe_gate_loss": "15.642", "overflow_expert1": "16.52", "overflow_expert2": "59.608", "entropy_gating": "1.983", "expert1_balance_top": "63.168", "expert1_balance_bottom": "1.88", "unused_expert1_count": "0.564", "expert2_balance_top": "49.929", "expert2_balance_bottom": "3.712", "unused_expert2_count": "0.368", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "15.969", "ppl": "64132.9", "wps": "16591.9", "ups": "0.25", "wpb": "65536", "bsz": "32", "num_updates": "20", "lr": "1.4e-05", "gnorm": "1.82", "loss_scale": "16", "train_wall": "39", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "19", "cuda_gb_free": "21.4", "wall": "107"}
2021-08-13 15:00:52 | INFO | train_inner | {"epoch": 1, "update": 0.011, "loss": "15.132", "moe_gate_loss": "13.3857", "overflow_expert1": "5.742", "overflow_expert2": "45.276", "entropy_gating": "2.023", "expert1_balance_top": "49.599", "expert1_balance_bottom": "5.064", "unused_expert1_count": "0.423", "expert2_balance_top": "40.013", "expert2_balance_bottom": "7.728", "unused_expert2_count": "0.32", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "14.939", "ppl": "31410.7", "wps": "16562", "ups": "0.25", "wpb": "65536", "bsz": "32", "num_updates": "30", "lr": "2.06667e-05", "gnorm": "1.397", "loss_scale": "16", "train_wall": "40", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "19", "cuda_gb_free": "21.4", "wall": "147"}
```

**Sample output on 8 x A100:**
```
2021-08-13 14:58:39 | INFO | fairseq.modules.fused_bias_gelu | Done with compiling and loading fused kernels.
2021-08-13 22:10:38 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 64.0
2021-08-13 22:10:40 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 32.0
2021-08-13 22:10:43 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 16.0
2021-08-13 22:11:02 | INFO | train_inner | {"epoch": 1, "update": 0.004, "loss": "20.703", "moe_gate_loss": "16.7792", "overflow_expert1": "21.27", "overflow_expert2": "52.991", "entropy_gating": "1.943", "expert1_balance_top": "66.899", "expert1_balance_bottom": "2.586", "unused_expert1_count": "0.13", "expert2_balance_top": "50.174", "expert2_balance_bottom": "5.421", "unused_expert2_count": "0.066", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "20.461", "ppl": "1.44332e+06", "wps": "34799.2", "ups": "0.53", "wpb": "65536", "bsz": "32", "num_updates": "10", "lr": "7.33333e-06", "gnorm": "29.972", "loss_scale": "16", "train_wall": "49", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "20.6", "cuda_gb_free": "29.2", "wall": "68"}
2021-08-13 22:11:21 | INFO | train_inner | {"epoch": 1, "update": 0.007, "loss": "16.195", "moe_gate_loss": "15.6466", "overflow_expert1": "16.589", "overflow_expert2": "59.311", "entropy_gating": "1.984", "expert1_balance_top": "63.15", "expert1_balance_bottom": "1.885", "unused_expert1_count": "0.548", "expert2_balance_top": "49.952", "expert2_balance_bottom": "3.785", "unused_expert2_count": "0.349", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "15.969", "ppl": "64151.2", "wps": "34973.5", "ups": "0.53", "wpb": "65536", "bsz": "32", "num_updates": "20", "lr": "1.4e-05", "gnorm": "1.822", "loss_scale": "16", "train_wall": "19", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "20.6", "cuda_gb_free": "29.2", "wall": "87"}
2021-08-13 22:11:39 | INFO | train_inner | {"epoch": 1, "update": 0.011, "loss": "15.131", "moe_gate_loss": "13.3747", "overflow_expert1": "5.894", "overflow_expert2": "44.769", "entropy_gating": "2.024", "expert1_balance_top": "49.877", "expert1_balance_bottom": "5.076", "unused_expert1_count": "0.41", "expert2_balance_top": "39.921", "expert2_balance_bottom": "7.812", "unused_expert2_count": "0.343", "all_to_all_cpu_time_ms": "0", "all_to_all_cuda_time_ms": "0", "inner_loss": "14.938", "ppl": "31389.2", "wps": "35046.4", "ups": "0.53", "wpb": "65536", "bsz": "32", "num_updates": "30", "lr": "2.06667e-05", "gnorm": "1.396", "loss_scale": "16", "train_wall": "19", "cuda_gb_allocated": "10.4", "cuda_gb_reserved": "20.6", "cuda_gb_free": "29.2", "wall": "105"}
```

## Larger model on multiple nodes

The following command will train an MoE model with 142B parameters on 64 A100s.

```bash
# salloc command might look different
salloc --gpus-per-node 8 --ntasks-per-node 8 --cpus-per-task 12 --nodes 8 --mem-per-gpu 128G

# set NUM_EXPERTS based on # of GPUs and desired # experts per GPU
# generally it's recommended to have a single expert per GPU
NUM_EXPERTS=64
TOKENS_PER_SAMPLE=1024

# launch the job (adjust port and --cpu-bind if needed)
DISTRIBUTED_PORT=12345
srun --cpu-bind=mask_cpu:000000ffffff000000ffffff,000000ffffff000000ffffff,000000ffffff000000ffffff,000000ffffff000000ffffff,ffffff000000ffffff000000,ffffff000000ffffff000000,ffffff000000ffffff000000,ffffff000000ffffff000000 \
  python fairseq_cli/train.py \
  --distributed-port $DISTRIBUTED_PORT \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
  --task dummy_lm --tokens-per-sample $TOKENS_PER_SAMPLE \
  --arch transformer_lm_gpt --share-decoder-input-output-embed \
  --decoder-layers 32 --decoder-embed-dim 4096 --decoder-ffn-embed-dim 16384 \
  --decoder-attention-heads 32 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion moe_cross_entropy --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
  --optimizer adam --fp16-adam-stats --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --warmup-updates 750 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 12 --update-freq 1 \
  --max-update 250 --disable-validation \
  --log-format json --log-interval 10
```
