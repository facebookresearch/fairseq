# Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)
This page contains pointers to pre-trained models as well as instructions on how to train new models for [our paper](https://arxiv.org/abs/1901.10430)

## Citation:
```bibtex
@inproceedings{wu2018pay,
  title = {Pay Less Attention with Lightweight and Dynamic Convolutions},
  author = {Felix Wu and Angela Fan and Alexei Baevski and Yann Dauphin and Michael Auli},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
  url = {https://arxiv.org/abs/1901.10430},
}
```

## Translation

### Pre-trained models
For some datasets we release models without GLUs which are faster at inference.

Description | Dataset | Model | Test set(s)
---|---|---|---
LightConv (without GLUs) | [IWSLT14 German-English](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/iwslt14.de-en.lightconv.tar.bz2) | IWSLT14 test: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/iwslt14.de-en.test.tar.bz2)
DynamicConv (without GLUs) | [IWSLT14 German-English](https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/iwslt14.de-en.dynamicconv.tar.bz2) | IWSLT14 test: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/iwslt14.de-en.test.tar.bz2)
LightConv (without GLUs) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.lightconv.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
DynamicConv (without GLUs) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.dynamicconv.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
LightConv | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.lightconv-glu.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
DynamicConv | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt16.en-de.joined-dict.dynamicconv-glu.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
LightConv | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt14.en-fr.joined-dict.lightconv-glu.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
DynamicConv | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt14.en-fr.joined-dict.dynamicconv-glu.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
LightConv | [WMT17 Chinese-English](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.zh-en.lightconv-glu.tar.bz2) | newstest2017: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.zh-en.newstest2017.tar.bz2)
DynamicConv | [WMT17 Chinese-English](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.zh-en.dynamicconv-glu.tar.bz2) | newstest2017: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.zh-en.newstest2017.tar.bz2)
LightConv (CUDA module) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.en-de.joined-dict.transformer.light-conv-cuda-glu.tar.gz) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
DynamicConv (CUDA module) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.en-de.joined-dict.transformer.dynamic-conv-cuda-glu.tar.gz) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

### Memory-Efficient CUDA Kernels

Since the PyTorch implementations of Light/Dynamic conv are quite memory intensive, we have developed CUDA kernels that implement the light and dynamic convolution operator in a memory-efficient and performant manner. For large sequence lengths, these kernels save about 50% memory compared to the PyTorch equivalent. 

To install the kernels, use the commands below. Once installed, they will automatically be used in place of the PyTorch implementations whenever a light or dynamic convolution is used.

```sh
# to install lightconv
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install

# to install dynamicconv
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
```

### Preprocessing the training datasets

Please follow the instructions in [`examples/translation/README.md`](../translation/README.md) to preprocess the data.

### Training and evaluation options:
To use the model without GLU, please set `--encoder-glu 0 --decoder-glu 0`.
For LightConv, please use `--encoder-conv-type lightweight --decoder-conv-type lightweight`, otherwise the default is DynamicConv.
For best BLEU results, lenpen may need to be manually tuned.

To use the CUDA kernels, first install the PyTorch modules using the commands below
```sh
# to install lightconv
python fairseq/modules/lightconv_layer/cuda_function_gen.py
python fairseq/modules/lightconv_layer/setup.py install

# to install dynamicconv
python fairseq/modules/dynamicconv_layer/cuda_function_gen.py
python fairseq/modules/dynamicconv_layer/setup.py install
```
Once the CUDA modules are installed, they will automatically be used instead of the PyTorch modules.

### IWSLT14 De-En
Training and evaluating DynamicConv (without GLU) on a GPU:
```sh
# Training
SAVE="save/dynamic_conv_iwslt"
mkdir -p $SAVE 
CUDA_VISIBLE_DEVICES=0 $(which fairseq-train) data-bin/iwslt14.tokenized.de-en \
    --clip-norm 0 --optimizer adam --lr 0.0005 \
    --source-lang de --target-lang en --max-tokens 4000 --no-progress-bar \
    --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --ddp-backend=no_c10d \
    --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
    -a lightconv_iwslt_de_en --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 0 --decoder-glu 0
python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt14.tokenized.de-en --path "${SAVE}/checkpoint_last10_avg.pt" --batch-size 128 --beam 4 --remove-bpe --lenpen 1 --gen-subset test --quiet 
```

### WMT16 En-De
Training and evaluating DynamicConv (with GLU) on WMT16 En-De using cosine scheduler on one machine with 8 V100 GPUs:
```sh
# Training
SAVE="save/dynamic_conv_wmt16en2de"
mkdir -p $SAVE
python -m torch.distributed.launch --nproc_per_node 8 $(which fairseq-train) \
    data-bin/wmt16_en_de_bpe32k --fp16  --log-interval 100 --no-progress-bar \
    --max-update 30000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 16 --attention-dropout 0.1 --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 3584 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.001 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 20000 \
    --arch lightconv_wmt_en_de_big --save-dir $SAVE \
    --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 1 --decoder-glu 1

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt16.en-de.joined-dict.newstest2014 --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 5 --remove-bpe --lenpen 0.5 --gen-subset test > wmt16_gen.txt
bash scripts/compound_split_bleu.sh wmt16_gen.txt
```

### WMT14 En-Fr
Training DynamicConv (with GLU) on WMT14 En-Fr using cosine scheduler on one machine with 8 V100 GPUs:
```sh
# Training
SAVE="save/dynamic_conv_wmt14en2fr"
mkdir -p $SAVE
python -m torch.distributed.launch --nproc_per_node 8 $(which fairseq-train) \
    data-bin/wmt14_en_fr --fp16  --log-interval 100 --no-progress-bar \
    --max-update 30000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 16 --attention-dropout 0.1 --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 3584 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.001 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 70000 \
    --arch lightconv_wmt_en_fr_big --save-dir $SAVE \
    --dropout 0.1 --attention-dropout 0.1 --weight-dropout 0.1 \
    --encoder-glu 1 --decoder-glu 1

# Evaluation
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14.en-fr.joined-dict.newstest2014 --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 5 --remove-bpe --lenpen 0.9 --gen-subset test
```
