# fairseq-py
FAIR Sequence-to-Sequence (PyTorch)

## Setup

Build PyTorch from source:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# install to ~/local/miniconda3

mkdir -p ~/src
CUDNN_VERSION=6.0  # or CUDNN_VERSION=5.1
cp -r ~myleott/src/cudnn-8.0-linux-x64-v$CUDNN_VERSION ~/src/

mkdir -p ~/local/miniconda3/etc/conda/activate.d
echo "export CMAKE_PREFIX_PATH=~/local/miniconda3" >> ~/local/miniconda3/etc/conda/activate.d/set_env.sh
echo "export LD_LIBRARY_PATH=~/src/cudnn-8.0-linux-x64-v$CUDNN_VERSION/cuda/lib64:~/local/miniconda3/lib:$LD_LIBRARY_PATH" >> ~/local/miniconda3/etc/conda/activate.d/set_env.sh
echo "export CUDNN_INCLUDE_DIR=~/src/cudnn-8.0-linux-x64-v$CUDNN_VERSION/cuda/include" >> ~/local/miniconda3/etc/conda/activate.d/set_env.sh
echo "export CUDNN_LIB_DIR=~/src/cudnn-8.0-linux-x64-v$CUDNN_VERSION/cuda/lib64" >> ~/local/miniconda3/etc/conda/activate.d/set_env.sh

. ~/local/miniconda3/bin/activate
conda install gcc mkl numpy
conda install magma-cuda80 -c soumith
pip install cmake --proxy=http://fwdproxy:8080/

# make sure you've configured proxy settings for git:
# https://our.intern.facebook.com/intern/wiki/Development_Environment/Internet_Proxy/#Git

cd ~/src
git clone https://github.com/pytorch/pytorch.git
cd pytorch
pip install -r requirements.txt --proxy=http://fwdproxy:8080/
python setup.py install
```

## Training examples

To train IWSLT de-en:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/sgross/fairseq/iwslt14_de-en/
python train.py $DATADIR -a fconv_iwslt_de_en --lr 0.25 --clip-norm 0.1 --dropout 0.2
```

To train WMT'16 en-ro:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/sgross/fairseq/wmt16_en2ro/
python train.py $DATADIR -a fconv_wmt_en_ro --lr 0.25 --clip-norm 0.1 --dropout 0.1 -b 64 --max-len 1600
```
