# Beyond English-Centric Multilingual Machine Translation

## Introduction
In this work, we create a true Many-to-Many multilingual translation model that can translate directly between any pair of 100 languages. Our focus on non-English-Centric models brings gains of more than 10 BLEU when directly translating between non-English directions while performing competitively with the best single systems of WMT. 

If you are new to using fairseq, read the following walkthrough. Otherwise, skip to the sections below. 

0. **Generation Data**

To download the generation data, follow the below commands. Note that all datasets need to be detokenized *before* applying SPM in the data preprocessing step. If you use these evaluation datasets, please cite their associated papers. 
```bash
# WMT - use sacrebleu, example here:
sacrebleu -t wmt14 -l fr-en --echo src > wmt.test.fr-en.fr
sacrebleu -t wmt14 -l fr-en --echo ref > wmt.test.fr-en.en

# WAT
wget http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/wat2020.my-en.zip
unzip wat2020.my-en.zip

# FLORES
# download from: https://github.com/facebookresearch/flores

# TED - need to detokenize with Moses!
# from: https://github.com/neulab/word-embeddings-for-nmt
wget http://phontron.com/data/ted_talks.tar.gz

# Autshumato
# request to download: https://repo.sadilar.org/handle/20.500.12185/397

# Tatoeba Challenge
# available here: https://github.com/Helsinki-NLP/Tatoeba-Challenge
```

1. **Training Data**

To produce the training data, we use a combination of [CCMatrix](https://arxiv.org/abs/1911.04944) and [CCAligned](https://arxiv.org/abs/1911.06154). Check out the instructions [here](https://github.com/facebookresearch/LASER/tree/master/tasks/CCMatrix) to download the raw data.

2. **Preprocess Data**

After downloading raw data, you will need to postprocess the data, then apply SPM, then binarize. Note that it is very important you run the postprocessing script, because this removes any instance of the evaluation data in the mined training data.

```bash
# preprocess data

# remove sentences with more than 50% punctuation
python /path/to/fairseq/examples/m2m_100/process_data/remove_too_much_punc.py 

# deduplicate training data
paste /path/to/datadir/train.$src /path/to/datadir/train.$tgt | awk '!x[$0]++' > /path/to/datadir/train.dedup
echo "keeping $(wc -l /path/to/datadir/train.dedup) bitext out of $(wc -l /path/to/datadir/train.$src)"
cut -f1 /path/to/datadir/train.dedup > /path/to/datadir/train.$src
cut -f2 /path/to/datadir/train.dedup > /path/to/datadir/train.$tgt

# remove all instances of evaluation data from the training data
python /path/to/fairseq/examples/m2m_100/process_data/dedup_data.py 

# frequency cleaning
wget https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz 
tar -xvzf histograms.tar.gz
python /path/to/fairseq/examples/m2m_100/process_data/clean_histogram.py --src $src --tgt $tgt --src-file /path/to/source/file --tgt-file /path/to/output/file --src-output-file source_output.$src --tgt-output-file target_output.$tgt --histograms /path/to/histograms

# apply SPM
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
python /path/to/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=/path/to/input/file/here \
    --outputs=/path/to/output/file/here

# length ratio cleaning
perl mosesdecoder/scripts/training/clean-corpus-n.perl --ratio 3 /path/to/training/data/train.spm.$src-$tgt $src $tgt /path/to/output/directory/train.spm.$src-$tgt 1 250

# binarize data
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --testpref spm.$src.$tgt \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
```

3. **Training Scripts**

To reproduce the training of our models, we train with fairseq-py's multilingual translation [task](https://github.com/pytorch/fairseq/tree/master/examples/multilingual). If you are interested in model parallel training, also check out [fairscale](https://github.com/facebookresearch/fairscale).

4. **Generation**

To generate from our models, follow the the commands in the generation section below.


If you use any of the resources listed here, please cite:
```bibtex
@article{fan2020beyond,
  title={Beyond English-Centric Multilingual Machine Translation},
  author={Fan, Angela and Bhosale, Shruti and Schwenk, Holger and Ma, Zhiyi and El-Kishky, Ahmed and Goyal, Siddharth and Baines, Mandeep and Celebi, Onur and Wenzek, Guillaume and Chaudhary, Vishrav and Goyal, Naman and Birch, Tom and Liptchinsky, Vitaliy and Edunov, Sergey and Grave, Edouard and Auli, Michael and Joulin, Armand},
  journal={arXiv preprint},
  year={2020}
}

@article{schwenk2019ccmatrix,
  title={Ccmatrix: Mining billions of high-quality parallel sentences on the web},
  author={Schwenk, Holger and Wenzek, Guillaume and Edunov, Sergey and Grave, Edouard and Joulin, Armand},
  journal={arXiv preprint arXiv:1911.04944},
  year={2019}
}

@article{el2019massive,
  title={A Massive Collection of Cross-Lingual Web-Document Pairs},
  author={El-Kishky, Ahmed and Chaudhary, Vishrav and Guzman, Francisco and Koehn, Philipp},
  journal={arXiv preprint arXiv:1911.06154},
  year={2019}
}
```


## Trained Models

### 418M and 1.2B Model
We include the last checkpoint for both of these models. 

```bash
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt 

# 418M parameter model
wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt 

# 1.2B parameter model
wget https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt

# Generation:
fairseq-generate $binarized_data_path --batch-size 32 --path $path_to_model --fixed-dictionary model_dict.128k.txt -s en -t fr --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test > gen_out
```

### 12B Model
12B parameter model trained on many-to-many training data for 100 languages. We include the last checkpoint, average of last 5 checkpoints, average of last 10 checkpoints. There isn't a universally best choice out of these three, but all three versions are pretty close in accuracy. You can either sweep over the 3 checkpoints on a dev test and use the best performing checkpoint for final testing. Or the last checkpoint can be a good default choice.

**Model Download Links**
Configuration | 2 32GB GPUs | 4 16GB GPUs | 6 12GB GPUs | 8 8GB GPUs
:--|:--|:--|:--|:--
Last Checkpoint | [12b_last_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_2_gpus.pt) | [12b_last_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_4_gpus.pt) | [12b_last_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_6_gpus.pt) | [12b_last_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_8_gpus.pt)
Average of last 5 checkpoints | [12b_avg5_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_2_gpus.pt) | [12b_avg5_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_4_gpus.pt) | [12b_avg5_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_6_gpus.pt) | [12b_avg5_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg5_chk_8_gpus.pt)
Average of last 10 checkpoints |  [12b_avg10_chk_2_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_2_gpus.pt) | [12b_avg10_chk_4_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_4_gpus.pt) | [12b_avg10_chk_6_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_6_gpus.pt) | [12b_avg10_chk_8_gpus.pt](https://dl.fbaipublicfiles.com/m2m_100/12b_avg10_chk_8_gpus.pt)

**Generation Arguments**
Configuration | 2 32GB GPUs | 4 16GB GPUs | 6 12GB GPUs | 8 8GB GPUs
:--|:--|:--|:--|:--
`--pipeline-encoder-balance` | `[26]` | `[1,15,10]` | `[1,9,9,7]` | `[1,6,6,6,7]`
`--pipeline-encoder-devices` | `[0]` | `[0,1,0]` | `[0,1,2,0]` | `[0,4,5,1,0]`
`--pipeline-decoder-balance` | `[3,22,1]` | `[3,11,11,1]` | `[3,7,7,8,1]` | `[1,6,6,6,6,1]`
`--pipeline-decoder-devices` | `[0,1,0]` | `[0,2,3,0]` | `[0,3,4,5,0]` |  `[0,2,6,7,3,0]`


## SentencePiece Model

```bash
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
```

## Generation with M2M-100

### Encode using our SentencePiece Model

Note: Install SentencePiece from [here](https://github.com/google/sentencepiece)

```bash
fairseq=/path/to/fairseq
cd $fairseq
sacrebleu --echo src -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.de
sacrebleu --echo ref -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.fr
wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
for lang in de fr ; do
    python scripts/spm_encode.py \
        --model spm.128k.model \
        --output_format=piece \
        --inputs=raw_input.de-fr.${lang} \
        --outputs=spm.de-fr.${lang}
done
```

### Binarization

```bash
wget https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt
fairseq-preprocess \
    --source-lang de --target-lang fr \
    --testpref spm.de-fr \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
```

### Generation for the 12B model

Note that generation can currently be run using 2 32GB / 4 16GB / 6 12GB / 8 8GB GPUs, and the corresponding model checkpoints and pipeline arguments can be found in the [12B Model Section](#12b-model).
Generation on CPUs will be added in the future.

```bash
wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs.txt
wget https://dl.fbaipublicfiles.com/m2m_100/12b_last_chk_4_gpus.pt
fairseq-generate \
    data_bin \
    --batch-size 1 \
    --path 12b_last_chk_4_gpus.pt \
    --fixed-dictionary model_dict.128k.txt \
    -s de -t fr \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs language_pairs.txt \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn \
    --pipeline-model-parallel \
    --pipeline-chunks 1 \
    --pipeline-encoder-balance '[1,15,10]' \
    --pipeline-encoder-devices '[0,1,0]' \
    --pipeline-decoder-balance '[3,11,11,1]' \
    --pipeline-decoder-devices '[0,2,3,0]' > gen_out
```
## Evaluation with M2M-100

### Tokenization

Note: Refer to tokenizers/README.md for more details on tokenization.

```bash
cd ${fairseq}/examples/m2m_100
cat ${fairseq}/gen_out | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh fr > hyp
cat ${fairseq}/raw_input.de-fr.fr | sh tok.sh fr > ref
```

### BLEU

```bash
sacrebleu -tok 'none' ref < hyp
```
