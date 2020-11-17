# Language Models not just for Pre-training: Fast Online Neural Noisy Channel Modeling

## Introduction
- [Yee et al. (2019)](https://www.aclweb.org/anthology/D19-1571.pdf) introduce a simple and effective noisy channel modeling approach for neural machine translation. However, the noisy channel online decoding approach introduced in this paper is too slow to be practical.
- To address this, [Bhosale et al. (2020)](http://www.statmt.org/wmt20/pdf/2020.wmt-1.68.pdf) introduces 3 simple approximations to make this approach very fast and practical without much loss in accuracy.
- This README provides intructions on how to run online decoding or generation with the noisy channel modeling approach, including ways to make it very fast without much loss in accuracy.

## Noisy Channel Modeling

[Yee et al. (2019)](https://www.aclweb.org/anthology/D19-1571.pdf) applies the Bayes Rule to predict `P(y|x)`, the probability of the target `y` given the source `x`.
```P(y|x) = P(x|y) * P(y) / P(x)```
- `P(x|y)` predicts the source `x` given the target `y` and is referred to as the **channel model**
- `P(y)` is a **language model** over the target `y`
- `P(x)` is generally not modeled since it is constant for all `y`.

We use Transformer models to parameterize the direct model `P(y|x)`, the channel model `P(x|y)` and the language model `P(y)`.

During online decoding with beam search, we generate the top `K2` candidates per beam and score them with the following linear combination of the channel model, the language model as well as the direct model scores.

```(1 / t) * log(P(y|x) + (1 / s) * ( λ1 * log(P(x|y)) + λ2 * log(P(y) ) )```
- `t` - Target Prefix Length
- `s` - Source Length
- `λ1` - Channel Model Weight
- `λ2` - Language Model Weight

The top `beam_size` candidates based on the above combined scores are chosen to continue the beams in beam search. In beam search with a direct model alone, the scores from the direct model `P(y|x)` are used to choose the top candidates in beam search.

This framework provides a great way to utlize strong target language models trained on large amounts of unlabeled data. Language models can prefer targets unrelated to the source, so we also need a channel model whose role is to ensure that the target preferred by the language model also translates back to the source.

### Training Translation Models and Language Models

For training Transformer models in fairseq for machine translation, refer to instructions [here](https://github.com/pytorch/fairseq/tree/master/examples/translation)

For training Transformer models in fairseq for language modeling, refer to instructions [here](https://github.com/pytorch/fairseq/tree/master/examples/language_model)

### Generation with Language Model for German-English translation with fairseq

Here are instructions to generate using a direct model and a target-side language model.

Note:
- Download and install fairseq as per instructions [here](https://github.com/pytorch/fairseq)
- Preprocess and binarize the dataset as per instructions in section [Test Data Preprocessing](#test-data-preprocessing)

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt

k2=10
lenpen=0.16
lm_wt=0.14
fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --k2 ${k2} \
    --combine-method lm_only \
    --task noisy_channel_translation \
    --lenpen ${lenpen} \
    --lm-wt ${lm_wt} \
    --gen-subset valid \
    --remove-bpe \
    --fp16 \
    --batch-size 10
```
### Noisy Channel Generation for German-English translation with fairseq

Here are instructions for noisy channel generation with a direct model, channel model and language model as explained in section [Noisy Channel Modeling](#noisy-channel-modeling).

Note:
- Download and install fairseq as per instructions [here](https://github.com/pytorch/fairseq)
- Preprocess and binarize the dataset as per instructions in section [Test Data Preprocessing](#test-data-preprocessing)

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
ch_model=en_de.big.seed4.pt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big.seed4.pt -O ${ch_model}

k2=10
lenpen=0.21
lm_wt=0.50
bw_wt=0.30
fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --channel-model ${ch_model} \
    --k2 ${k2} \
    --combine-method noisy_channel \
    --task noisy_channel_translation \
    --lenpen ${lenpen} \
    --lm-wt ${lm_wt} \
    --ch-wt ${bw_wt} \
    --gen-subset test \
    --remove-bpe \
    --fp16 \
    --batch-size 1
```
## Fast Noisy Channel Modeling

[Bhosale et al. (2020)](http://www.statmt.org/wmt20/pdf/2020.wmt-1.68.pdf) introduces 3 approximations that speed up online noisy channel decoding -
- Smaller channel models (`Tranformer Base` with 1 encoder and decoder layer each vs. `Transformer Big`)
  - This involves training a channel model that is possibly smaller and less accurate in terms of BLEU than a channel model of the same size as the direct model.
  - Since the role of the channel model is mainly to assign low scores to generations from the language model if they don't translate back to the source, we may not need the most accurate channel model for this purpose.
- Smaller output vocabulary size for the channel model (~30,000 -> ~1000)
  - The channel model doesn't need to score the full output vocabulary, it just needs to score the source tokens, which are completely known.
  - This is specified using the arguments `--channel-scoring-type src_vocab --top-k-vocab 500`
  - This means that the output vocabulary for the channel model will be the source tokens for all examples in the batch and the top-K most frequent tokens in the vocabulary
  - This reduces the memory consumption needed to store channel model scores significantly
- Smaller number of candidates (`k2`) scored per beam
  - This is specified by reducing the argument `--k2`


### Fast Noisy Channel Generation for German-English translation with fairseq

Here are instructions for **fast** noisy channel generation with a direct model, channel model and language model as explained in section [Fast Noisy Channel Modeling](#fast-noisy-channel-modeling). The main differences are that we use a smaller channel model, reduce `--k2`, set `--channel-scoring-type src_vocab --top-k-vocab 500` and increase the `--batch-size`.

Note:
- Download and install fairseq as per instructions [here](https://github.com/pytorch/fairseq)
- Preprocess and binarize the dataset as per instructions in section [Test Data Preprocessing](#test-data-preprocessing)

```sh
binarized_data=data_dir/binarized
direct_model=de_en_seed4.pt
lm_model=en_lm.pt
lm_data=lm_data
small_ch_model=en_de.base_1_1.seed4.pt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt -O ${direct_model}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt -O ${lm_model}
mkdir -p ${lm_data}
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/dict.txt -O ${lm_data}/dict.txt
wget https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base_1_1.seed4.pt -O ${small_ch_model}

k2=3
lenpen=0.23
lm_wt=0.58
bw_wt=0.26
fairseq-generate ${binarized_data} \
    --user-dir examples/fast_noisy_channel \
    --beam 5 \
    --path ${direct_model} \
    --lm-model ${lm_model} \
    --lm-data ${lm_data}  \
    --channel-model ${small_ch_model} \
    --k2 ${k2} \
    --combine-method noisy_channel \
    --task noisy_channel_translation \
    --lenpen ${lenpen} \
    --lm-wt ${lm_wt} \
    --ch-wt ${bw_wt} \
    --gen-subset test \
    --remove-bpe \
    --fp16 \
    --batch-size 50 \
    --channel-scoring-type src_vocab --top-k-vocab 500
```

## Test Data Preprocessing

For preprocessing and binarizing the test sets for Romanian-English and German-English translation, we use the following script -

```sh
FAIRSEQ=/path/to/fairseq
cd $FAIRSEQ
SCRIPTS=$FAIRSEQ/mosesdecoder/scripts
if [ ! -d "${SCRIPTS}" ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORMALIZE=$SCRIPTS/tokenizer/normalize-punctuation.perl

s=de
t=en
test=wmt18

mkdir -p data_dir

# Tokenization
if [ $s == "ro" ] ; then
    # Note: Get normalise-romanian.py and remove-diacritics.py from
    # https://github.com/rsennrich/wmt16-scripts/tree/master/preprocess
    sacrebleu -t $test -l $s-$t --echo src | \
        $NORMALIZE -l $s | \
        python normalise-romanian.py | \
        python remove-diacritics.py | \
        $TOKENIZER -l $s -a -q > data_dir/$test.$s-$t.$s
else
    sacrebleu -t $test -l $s-$t --echo src | perl $NORMALIZE -l $s | perl $TOKENIZER -threads 8 -a -l $s > data_dir/$test.$s-$t.$s
fi

sacrebleu -t $test -l $s-$t --echo ref | perl $NORMALIZE -l $t | perl $TOKENIZER -threads 8 -a -l $t > data_dir/$test.$s-$t.$t


# Applying BPE
src_bpe_code=/path/to/source/language/bpe/code
tgt_bpe_code=/path/to/target/language/bpe/code
src_dict=/path/to/source/language/dict
tgt_dict=/path/to/target/language/dict

FASTBPE=$FAIRSEQ/fastBPE
if [ ! -d "${FASTBPE}" ] ; then
    git clone https://github.com/glample/fastBPE.git
    # Follow compilation instructions at https://github.com/glample/fastBPE
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi

${FASTBPE}/fast applybpe data_dir/bpe.$test.$s-$t.$s data_dir/$test.$s-$t.$s ${src_bpe_code}
${FASTBPE}/fast applybpe data_dir/bpe.$test.$s-$t.$s data_dir/$test.$s-$t.$s ${tgt_bpe_code}

fairseq-preprocess -s $s -t $t \
    --testpref data_dir/bpe.$test.$s-$t \
    --destdir data_dir/binarized \
    --srcdict ${src_dict} \
    --tgtdict ${tgt_dict}
```

## Calculating BLEU

```sh
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
cat ${generation_output} | grep -P "^H" | sort -V | cut -f 3- | $DETOKENIZER -l $t -q -a | sacrebleu -t $test -l $s-$t
```


## Romanian-English Translation

The direct and channel models are trained using bitext data (WMT16) combined with backtranslated data (The monolingual data used for backtranslation comes from http://data.statmt.org/rsennrich/wmt16_backtranslations/ (Sennrich et al., 2016c))

The backtranslated data is generated using an ensemble of 3 English-Romanian models trained on bitext training data (WMT16) with unrestricted sampling.

### BPE Codes and Dictionary

We learn a joint BPE vocabulary of 18K types on the bitext training data which is used for both the source and target.
||Path|
|----------|------|
| BPE Code | [joint_bpe_18k](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/bpe_18k) |
| Dictionary | [dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/dict) |

### Direct Models
For Ro-En with backtranslation, the direct and channel models use a Transformer-Big architecture.

| Seed | Model |
|----|----|
| 2 | [ro_en_seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed2.pt)
| 4 | [ro_en_seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed4.pt)
| 6 | [ro_en_seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/direct_models/seed6.pt)

### Channel Models
For channel models, we follow the same steps as for the direct models. But backtranslated data is generated in the opposite direction using [this Romanian monolingual data](http://data.statmt.org/rsennrich/wmt16_backtranslations/).
The best lenpen, LM weight and CH weight are obtained by sweeping over the validation set (wmt16/dev) using beam 5.
| Model Size | Lenpen | LM Weight | CH Weight | Seed 2 | Seed 4 | Seed 6 |
|----|----|----|----|----|----|----|
| `big` | 0.84 | 0.64 | 0.56 | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) | [big.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/big.seed2.pt) |
| `base_1_1` | 0.63 | 0.40 | 0.37 | [base_1_1.seed2.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed2.pt) | [base_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed4.pt) | [base_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/channel_models/base_1_1.seed6.pt) |

### Language Model
The model is trained on de-duplicated English Newscrawl data from 2007-2018 comprising 186 million sentences or 4.5B words after normalization and tokenization.
|  | Path |
|----|----|
| `--lm-model` | [transformer_en_lm](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/lm_model/transformer_lm.pt) |
| `--lm-data` | [lm_data](https://dl.fbaipublicfiles.com/fast_noisy_channel/ro_en/lm_model/lm_dict)

## German-English Translation

### BPE Codes and Dictionaries

| | Path|
|----------|------|
| Source BPE Code | [de_bpe_code_24K](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/de_bpe_code_24K) |
| Target BPE Code | [en_bpe_code_24K](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/en_bpe_code_24K)
| Source Dictionary | [de_dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/de_dict) |
| Target Dictionary | [en_dict](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/en_dict) |

### Direct Models
We train on WMT’19 training data. Following [Ng et al., 2019](http://statmt.org/wmt19/pdf/53/WMT33.pdf), we apply language identification filtering and remove sentences longer than 250 tokens as well as sentence pairs with a source/target length ratio exceeding 1.5. This results in 26.8M sentence pairs.
We use the Transformer-Big architecture for the direct model.

| Seed | Model |
|:----:|----|
| 4 | [de_en_seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed4.pt)
| 5 | [de_en_seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed5.pt)
| 6 | [de_en_seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/direct_models/seed6.pt)

### Channel Models

We train on WMT’19 training data. Following [Ng et al., 2019](http://statmt.org/wmt19/pdf/53/WMT33.pdf), we apply language identification filtering and remove sentences longer than 250 tokens as well as sentence pairs with a source/target length ratio exceeding 1.5. This results in 26.8M sentence pairs.

| Model Size | Seed 4 | Seed 5 | Seed 6 |
|----|----|----|----|
| `big` | [big.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big.seed4.pt) | [big.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big.seed5.pt) | [big.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big.seed6.pt) |
| `big_1_1` | [big_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big_1_1.seed4.pt) | [big_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big_1_1.seed5.pt) | [big_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/big_1_1.seed6.pt) |
| `base` | [base.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base.seed4.pt) | [base.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base.seed5.pt) | [base.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base.seed6.pt) |
| `base_1_1` | [base_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base_1_1.seed4.pt) | [base_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base_1_1.seed5.pt) | [base_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/base_1_1.seed6.pt) |
| `half` | [half.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half.seed4.pt) | [half.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half.seed5.pt) | [half.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half.seed6.pt) |
| `half_1_1` | [half_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half_1_1.seed4.pt) | [half_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half_1_1.seed5.pt) | [half_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/half_1_1.seed6.pt) |
| `quarter` | [quarter.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter.seed4.pt) | [quarter.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter.seed5.pt) | [quarter.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter.seed6.pt) |
| `quarter_1_1` | [quarter_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter_1_1.seed4.pt) | [quarter_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter_1_1.seed5.pt) | [quarter_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/quarter_1_1.seed6.pt) |
| `8th` | [8th.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th.seed4.pt) | [8th.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th.seed5.pt) | [8th.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th.seed6.pt) |
| `8th_1_1` | [8th_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th_1_1.seed4.pt) | [8th_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th_1_1.seed5.pt) | [8th_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/8th_1_1.seed6.pt) |
| `16th` | [16th.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th.seed4.pt) | [16th.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th.seed5.pt) | [16th.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th.seed6.pt) |
| `16th_1_1` | [16th_1_1.seed4.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th_1_1.seed4.pt) | [16th_1_1.seed5.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th_1_1.seed5.pt) | [16th_1_1.seed6.pt](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/channel_models/16th_1_1.seed6.pt) |

### Language Model
The model is trained on de-duplicated English Newscrawl data from 2007-2018 comprising 186 million sentences or 4.5B words after normalization and tokenization.
|  | Path |
|----|----|
| `--lm-model` | [transformer_en_lm](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/transformer_lm.pt) |
| `--lm-data` | [lm_data](https://dl.fbaipublicfiles.com/fast_noisy_channel/de_en/lm_model/lm_dict/)


## Citation

```bibtex
@inproceedings{bhosale2020language,
    title={Language Models not just for Pre-training: Fast Online Neural Noisy Channel Modeling},
    author={Shruti Bhosale and Kyra Yee and Sergey Edunov and Michael Auli},
    booktitle={Proceedings of the Fifth Conference on Machine Translation (WMT)},
    year={2020},
}

@inproceedings{yee2019simple,
  title={Simple and Effective Noisy Channel Modeling for Neural Machine Translation},
  author={Yee, Kyra and Dauphin, Yann and Auli, Michael},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={5700--5705},
  year={2019}
}
```
