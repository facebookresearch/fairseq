# Installation Guide for NLLB

## Install PyTorch

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## Install Apex

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

## Install Megatron

```bash
git clone --depth=1 --branch v2.4 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

## Install fairscale

```bash
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
# needed when loading MoE checkpoint w/num_experts < num_gpus
git checkout origin/experts_lt_gpus_moe_reload_fix
pip install -e .
```

## Install fairseq nllb branch

```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout nllb
pip install -e .
python setup.py build_ext --inplace
```

## Install stopes
```bash
git clone https://github.com/facebookresearch/stopes.git
cd stopes
pip install -e '.[dev]'
```

## Install pre-commit hooks

```bash
# turn on pre-commit hooks
pip install pre-commit && pre-commit install
```
