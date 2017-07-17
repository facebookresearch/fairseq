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

To prepare data and dictionary for IWSLT de-en:

```
TEXTROOT=/mnt/vol/gfsai-flash-east/ai-group/users/edunov/data/neuralmt/iwslt14_mixer_text/
python preprocess.py --source-lang de --target-lang en \
--trainpref $TEXTROOT/train --validpref $TEXTROOT/valid \
--testpref $TEXTROOT/test --thresholdtgt 3 --thresholdsrc 3 \
--destdir /data/users/$USER/iwslt_de_en
```

To train IWSLT de-en:

```
DATADIR=/data/users/$USER/iwslt_de_en
THC_CACHING_ALLOCATOR=1 OMP_NUM_THREADS=1 /usr/local/bin/fry_gpulock 1 -- \
    python train.py $DATADIR -a fconv_iwslt_de_en --lr 0.25 --clip-norm 0.1 \
    --dropout 0.2 --max-tokens 10000
```

To prepare data and dictionary for WMT'14 en-fr:

```
TEXTROOT=/mnt/vol/gfsai-oregon/ai-group/datasets/text/wmt14/en-fr/175/ratio1.5/bpej40k/clean/
python preprocess.py --source-lang en --target-lang fr \
--trainpref $TEXTROOT/train --validpref $TEXTROOT/valid \
--testpref $TEXTROOT/newstest2014 --thresholdtgt 0 --thresholdsrc 0 \
--destdir /data/users/$USER/wmt14_en_fr
```

To train WMT'14 en-fr:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/edunov/data/neuralmt/wmt14_en_fr/
THC_CACHING_ALLOCATOR=1 OMP_NUM_THREADS=1 /usr/local/bin/fry_gpulock 1 -- \
    python train.py $DATADIR -a fconv_wmt_en_fr --lr 0.25 --clip-norm 0.1 \
    --dropout 0.2 --max-tokens 1000
```

To prepare data and dictionary for WMT'14 en-de:

```
TEXTROOT=/data/users/michaelauli/data/wmt14/en-de/stanford/bpej40k/
python preprocess.py --source-lang en --target-lang de \
--trainpref $TEXTROOT/train-split --validpref $TEXTROOT/valid-split \
--testpref $TEXTROOT/newstest2014 --thresholdtgt 0 --thresholdsrc 0 \
--destdir /data/users/$USER/wmt14_en_de
```

To train WMT'14 en-de:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/edunov/data/neuralmt/wmt14_en_de/
THC_CACHING_ALLOCATOR=1 OMP_NUM_THREADS=1 /usr/local/bin/fry_gpulock 1 -- \
    python train.py $DATADIR -a fconv_wmt_en_de --lr 0.25 --clip-norm 0.1 \
    --dropout 0.2 --max-tokens 1500
```

To train WMT'16 en-ro:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/sgross/fairseq/wmt16_en2ro/
python train.py $DATADIR -a fconv_wmt_en_ro --lr 0.25 --clip-norm 0.1 --dropout 0.1 -b 64 --max-len 1600
```
