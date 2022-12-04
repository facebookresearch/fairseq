[[Back]](..)

# Unified Speech-Text Pre-training for Speech Translation and Recognition

This directory contains the  pre-training recipes from paper ["Unified Speech-Text Pre-training for Speech Translation and Recognition"](https://arxiv.org/abs/2204.05409).

## Librispeech ASR Pre-training
### Prepare Data
#### Download files
#### Prepare pre-training data
-   Text to text task (T2T): prepare the binary data following the similar steps in [EN_DE Joint training](./ende-mustc.md). The source  data is presented as phomeme token sequence and the target  data is coded as subword tokens via SentencePiece. The text data is downloaded from [openslr](https://www.openslr.org/12)
-   Self-supervised speech learning task (SSL): The data is prepared as [wav2vec 2.0](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/README.md)
-   Speech to phoneme classification task (S2P): The tsv file contains 5 fields: "id",  "audio",   "n_frames",    "tgt_text",  and  "align". The tgt_text field is corresponding to the phoneme based representation of the speech data. "align" field contains the alignment information. The phoneme level forced alignment for the labelled speech data (i.e. Librispeech) can be obtained via [kaldi](http://kaldi-asr.org) or [MFA](https://montrealcorpustools.github.io/Montreal-Forced-Aligner/). The segmentation information is normalized to 0$\sim$1 for the whole utterance. The snapshot of the tsv file is below:
```
id  audio   n_frames    tgt_text    align
116-288045-0000 /librispeech/dev-other/116/288045/116-288045-0000.flac    170400  <sil> ▁AE1 Z AY1 ▁AH0 P R OW1 CH T ▁DH AH1 ▁S IH1 T IY0 <sil> AY1 ▁HH ER1 D ▁B EH1 L Z ▁R IH1 NG IH0 NG <sil> ▁AE1 N D AH0 ▁L IH1 T AH0 L ▁L EY1 T ER0 AY1 ▁F AW1 N D ▁DH AH0 ▁S T R IY1 T S ▁AH0 S T IH1 R ▁W IH0 TH ▁TH R AO1 NG Z ▁AH0 V ▁W EH1 L ▁D R EH1 S T ▁P IY1 P AH0 L ▁IH1 N ▁F AE1 M L IY0 ▁G R UW1 P S <sil> ▁W EH1 N D IH0 NG ▁DH EH1 R ▁W EY1 <sil> ▁HH IH1 DH ER0 ▁AH0 N D ▁TH IH1 DH ER0 <sil> 0.047977 0.056444 0.064911 0.075259 0.081844 0.089370 0.095014 0.104421 0.109125 0.111947 0.115710 0.120414 0.134525 0.141110 0.143932 0.174036 0.176858 0.190028 0.199436 0.207902 0.218250 0.224835 0.231421 0.242709 0.251176 0.257761 0.263405 0.268109 0.270931 0.290687 0.342427 0.349953 0.353716 0.356538 0.360301 0.363123 0.365945 0.368768 0.371590 0.376294 0.384760 0.394167 0.401693 0.409219 0.419567 0.430856 0.441204 0.444026 0.446849 0.449671 0.456256 0.463782 0.471308 0.477893 0.486359 0.491063 0.494826 0.501411 0.512700 0.517404 0.520226 0.534337 0.540922 0.545626 0.550329 0.559737 0.568203 0.583255 0.592662 0.600188 0.603951 0.611477 0.619003 0.624647 0.634055 0.639699 0.646284 0.653810 0.659454 0.664158 0.670743 0.682032 0.687676 0.692380 0.708373 0.713076 0.719661 0.729069 0.740357 0.744120 0.748824 0.752587 0.761994 0.770461 0.781750 0.790216 0.805268 0.808090 0.823142 0.832549 0.836312 0.840075 0.843838 0.851364 0.854186 0.857008 0.862653 0.878645 0.898401 0.901223 0.906867 0.913452 0.920038 0.926623 0.934149 0.939793 0.942615 0.945437 0.952023 0.957667 0.977422 1.000000

```
-   Speech to text task (S2T): The data preparation follow the steps in [EN_DE Joint training](./ende-mustc.md).

#### Prepare fine-tuning data:
We re-use the data from T2T and S2T tasks in the fine-tuning stage.

### Model Build
#### Pre-training
```
python train.py  $T2T_DATA \
    --save-dir $SAVE_PRE_PATH --user-dir examples/speech_text_joint_to_text --task speech_text_joint_denoising \
    --criterion speech_text_pretrain_cross_entropy --optimizer adam --weight-decay 0.01 --config-yaml config_s2p.yaml --config-s2s-yaml config.yaml --ddp-backend no_c10d \
    --lang-pairs pho-wrd --num-workers 4 --log-interval 500 --save-interval-updates 5000 --keep-interval-updates 1 --no-emb-update-unsup --report-accuracy --lr 0.001 --end-learning-rate 1e-06 \
    --lr-scheduler polynomial_decay --warmup-updates 10000 --total-num-update 800000 --update-freq 6 --validate-interval-updates 10000 --train-subset train \
    --valid-subset valid,valid_sup_speech,valid_sup_speech_s2s,valid_unsup_speech --dataset-impl mmap \
    --sup-speech-data $S2P_DATA_PATH --sup-speech-train-subset train_960.ali --sup-speech-valid-subset dev-clean.ali --sup-speech-s2s-data $S2T_DATA_PATH \
    --sup-speech-s2s-train-subset train --sup-speech-s2s-valid-subset dev-clean --unsup-speech-train-data $SSL_DATA_PATH/train.tsv --unsup-speech-valid-data $SSL_DATA_PATH/valid.tsv \
    --batch-size 200 --batch-size-valid 150 --max-source-positions 1024 --max-target-positions 1024 --max-text-tokens 3072 --max-speech-positions 600000 \
    --max-sample-size 750000 --min-sample-size 64000 --max-speech-tokens 750000 --max-tokens-valid 750000 --skip-invalid-size-inputs-valid-test \
    --unsupervised-speech-sample-ratio 3.0 --supervised-speech-sample-ratio 5 --supervised-speech-s2s-sample-ratio 5 --text-sample-ratio 1.0 --mask 0.3 --mask-random 0.1 \
    --mask-length span-poisson --speech-sup-mask-prob 0.3 --speech-unsup-mask-prob 0.7 --use-mask-whole-words --arch speech_text_pretrain_bart_base_stack \
    --no-scale-feature --activation-fn gelu --speech-extractor-mode default --stacked-encoder all --encoder-normalize-before --decoder-normalize-before \
    --encoder-learned-pos --decoder-learned-pos --dropout 0.1 --load-pretrained-mbart-encoder-from $BART --load-pretrained-mbart-decoder-from $BART
```
The current implementation also supports model pre-training without the forced alignment supervised data. In this case, CTC is used to optimize the S2P task. We need to do following changes for the setting:
1. options to be added
```
--use-sup-speech-ctc --criterion speech_text_pretrain_compound
```
2. options to be deleted
```
--same-data-update --criterion speech_text_pretrain_cross_entropy
```
However, we find the CTC based pre-training is still worse than the forced alignment based setting. It could be partially due to the inferior pre-training setting that we re-use the forced alignment based pre-training setting for the CTC based pre-training.

#### Fine-tuning
```
python train.py  $S2T_DATA_PATH \
    --save-dir $SAVE_FT_PATH  --num-workers 8 --task speech_text_joint_to_text --arch dualinputs2twavtransformer_base_stack \
    --user-dir examples/speech_text_joint_to_text --max-update 100000 --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0003 --update-freq 3 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy --guide-alpha 0.8 --label-smoothing 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --enc-grad-mult 2.0 --max-tokens 800000 --max-source-positions 800000 --max-tokens-text 10000 --max-positions-text 1024 --max-target-positions 1024 --no-scale-feature \
    --activation-fn gelu --load-pretrained-speech-text-encoder $SAVE_PRE_PATH/checkpoint_last.pt --load-pretrained-speech-text-decoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --encoder-normalize-before --decoder-normalize-before --speech-extractor-mode default --speech-mask-channel-length 64 --speech-mask-channel-prob 0.5 \
    --speech-mask-length 10 --speech-mask-prob 0.65 --text-sample-ratio 0.25 --mask-text-ratio 0.3 --mask-text-type random --parallel-text-data text_bin \
    --text-input-cost-ratio 0.5 --langpairs pho-wrd --update-mix-data --log-format json --max-tokens-valid 800000 --ddp-backend no_c10d --log-interval 500 \
    --config-yaml config.yaml --skip-invalid-size-inputs-valid-test --keep-last-epochs 50 --layernorm-embedding --encoder-learned-pos --decoder-learned-pos
```

### Evaluation
The last 10 epoch models from fine-tuning is conducted model average to get $FINAL_MODEL
```
python ./fairseq_cli/generate.py \
    $S2T_DATA_PATH \
    --task speech_text_joint_to_text \
    --max-tokens 800000  \
    --max-source-positions 800000 \
    --nbest 1 \
    --results-path $RESULTS_LOG \
    --batch-size 512 \
    --path $FINAL_MODEL \
    --gen-subset $SUBSET \
    --config-yaml config.yaml \
    --scoring wer \
    --beam 10 --lenpen 1.0 examples/speech_text_joint_to_text
```

### Results and models
| | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| WER| 2.0 | 4.4 | 2.1 |4.6 |

**Model Links**:
-   [config_s2p.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/pretrain/config_s2p.yaml): Config for S2P
-   [spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned/spm.model): Sentence Piece model
-   [src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned/src_dict.txt): Source Phoneme Dictionary
-   [tgt_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned/tgt_dict.txt): Target Sentence Piece Dictionary
-   [config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned/config.yaml): Config for S2T
-   [BART](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/pretrain/bart.pt): trained from Librispeech text data
-   [Joint Pre-trained model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/pretrain/checkpoint6.pt): model pre-trained with 960 hours Librispeech data (S2P, S2T) Librispeech text training data (T2T) and Librilight data (SSL)
-   [Fine-tuned model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/librispeech/finetuned/checkpoint_ave_10.pt): the pre-trained model is fined one 960 hours Librispeech speech and text data. (S2T + T2T)

## MuST-C
### Prepare Data
Compared with the ASR Librispeech ASR recipe, the differences are below:
-   Replace the speech data with corresponding MuST-C data
-   Parallel text data from WMT is replaced the Librispeech text data

### Model Build
#### Pre-training
EN-DE is used as an example
```
python train.py  $TXT_DATA \
    --save-dir $SAVE_PRE_PATH  --user-dir examples/speech_text_joint_to_text --task speech_text_joint_denoising --criterion speech_text_pretrain_cross_entropy --optimizer adam --weight-decay 0.01 \
    --config-yaml config_s2p.yaml --config-s2s-yaml config.yaml --ddp-backend no_c10d --lang-pairs-bitext en-fr --num-workers 4 --log-interval 500 --save-interval-updates 5000 --keep-interval-updates 1 \
    --no-emb-update-unsup --use-decoder-output-proj --report-accuracy --lr 0.001 --end-learning-rate 1e-06 --lr-scheduler polynomial_decay --warmup-updates 10000 --total-num-update 800000 \
    --update-freq 8 --validate-interval-updates 10000 --train-subset train --valid-subset valid_sup_speech,valid_sup_speech_s2s,valid_unsup_speech --dataset-impl mmap \
    --sup-speech-data $S2P_DATA_PATH --sup-speech-train-subset train --sup-speech-valid-subset dev --sup-speech-s2s-data $S2T_DATA_PATH --sup-speech-s2s-train-subset train \
    --sup-speech-s2s-valid-subset dev --unsup-speech-train-data $SSL_DATA_PATH/train.tsv --unsup-speech-valid-data $SSL_DATA_PATH/valid.tsv --batch-size 200 --batch-size-valid 100 \
    --max-source-positions 1024 --max-target-positions 1024 --max-text-tokens 2048 --max-speech-positions 600000 --max-sample-size 600000 --min-sample-size 64000 \
    --max-speech-tokens 600000 --max-tokens-valid 600000 --skip-invalid-size-inputs-valid-test --unsupervised-speech-sample-ratio 1.2 --supervised-speech-sample-ratio 10 \
    --supervised-speech-s2s-sample-ratio 10 --bitext-sample-ratio 0.5 --mask 0.3 --mask-random 0.1 --mask-length span-poisson --speech-sup-mask-prob 0.3 \
    --speech-unsup-mask-prob 0.7 --use-mask-whole-words --arch speech_text_pretrain_bart_base_stack --no-scale-feature --activation-fn gelu --speech-extractor-mode default \
    --stacked-encoder s2s --encoder-normalize-before --decoder-normalize-before --encoder-learned-pos --decoder-learned-pos --dropout 0.1 \
    --load-pretrained-mbart-encoder-from $EN_FR_NMT --load-pretrained-mbart-decoder-from $EN_FR_NMT
```
#### Fine-tuning
```
python train.py $S2T_DATA_PATH \
    --save-dir $SAVE_FT_PATH --num-workers 8 --task speech_text_joint_to_text --arch dualinputs2twavtransformer_base_stack --user-dir examples/speech_text_joint_to_text \
    --max-epoch 25 --update-mix-data --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0003 --update-freq 4 --clip-norm 10.0 --warmup-updates 20000 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy --guide-alpha 0.8 --attentive-cost-regularization 0.02 --enc-grad-mult 2.0 --label-smoothing 0.1 \
    --max-tokens 800000 --max-source-positions 800000 --max-tokens-text 10000 --max-positions-text 1024 --load-pretrained-speech-text-encoder $SAVE_PRE_PATH/checkpoint_last.pt \
    --load-pretrained-speech-text-decoder $SAVE_PRE_PATH/checkpoint_last.pt  --speech-mask-channel-length 64 --speech-mask-channel-prob 0.5 --speech-mask-length 10 \
    --speech-mask-prob 0.65 --text-sample-ratio 0.05 --mask-text-ratio 0.3 --mask-text-type random --parallel-text-data data-bin-wt --text-input-cost-ratio 0.5 \
    --langpairs en-fr --log-format json --max-tokens-valid 800000 --ddp-backend no_c10d --log-interval 100 --config-yaml config.yaml --skip-invalid-size-inputs-valid-test \
    --noise-token '▁NOISE' --keep-last-epochs 40 --layernorm-embedding --encoder-learned-pos --decoder-learned-pos --activation-fn gelu \
    --speech-extractor-mode default --max-target-positions 1024 --encoder-normalize-before --decoder-normalize-before
```

### Evaluation
The last 10 epoch models from fine-tuning is conducted model average to get $FINAL_MODEL
```
python fairseq_cli/generate.py \
    $S2T_DATA_PATH \
    --task speech_text_joint_to_text \
    --nbest 1 \
    --max-tokens 800000 \
    --max-source-positions 800000 \
    --results-path $RESULTS_LOG \
    --batch-size 512 \
    --path $FINAL_MODEL \
    --gen-subset $SUBSET \
    --config-yaml config.yaml \
    --scoring sacrebleu \
    --beam 10 --lenpen 1.0 examples/speech_text_joint_to_text
```


### Results and models
| | en-fr | en-es | en-de |
|---|---|---|---|
| BLEU| 39.7 | 33.2 |29.2 |


**Model Links**:
1.  DE
  - [de config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/config.yaml)
  - [de src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/src_dict.txt)
  - [de tgt_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/tgt_dict.txt)
  - [de spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/spm.model)
  - [de pre-trained nmt model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/nmt.pt)
  - [de pre-trained model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/checkpoint_pretraing.pt)
  - [de fine-tuned model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/de/checkpoint_finetune_ave10.pt)
2.  ES
  - [es config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/config.yaml)
  - [es src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/src_dict.txt)
  - [es tgt_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/tgt_dict.txt)
  - [es spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/spm.model)
  - [es pre-trained nmt model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/nmt.pt)
  - [es pre-trained model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/checkpoint_pretraing.pt)
  - [es fine-tuned model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/es/checkpoint_finetune_ave10.pt)
3.  FR
  - [fr config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/config.yaml)
  - [fr src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/src_dict.txt)
  - [fr tgt_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/tgt_dict.txt)
  - [fr spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/spm.model)
  - [fr pre-trained nmt model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/nmt.pt)
  - [fr pre-trained model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/checkpoint_pretraing.pt)
  - [fr fine-tuned model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/fr/checkpoint_finetune_ave10.pt)
4. [config_s2p.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/acl2022/must_c/config_s2p.yaml)
