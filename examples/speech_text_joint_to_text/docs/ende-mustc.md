[[Back]](..)

# Joint Speech Text Training for the MuST-C English to German Speech Translation task

Joint Training Baseline: it is based on paper ["A general multi-task learning framework to leverage text data for speech to text tasks"](https://arxiv.org/pdf/2010.11338.pdf)

Enhanced Joint Training: the joint training is enhanced with pre-trained models, cross attentive regularization and online knowledge distillation based on paper ["Improving Speech Translation by Understanding and Learning from the Auxiliary Text Translation Task"](https://research.fb.com/publications/improving-speech-translation-by-understanding-and-learning-from-the-auxiliary-text-translation-task)

## Prepare Data
#### Download files
-   Sentence piece model [spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/spm.model)
-   Dictionary [dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/dict.txt)
-   config [config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/config.yaml)
#### Prepare MuST-C data set
-   Please follow the data preparation in the [S2T example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md)
-   Convert source text under the "src_text" column in the tsv file into phoneme representation.
```bash
    python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
        --lower-case --do-filter --use-word-start --no-punc \
        --reserve-word examples/speech_text_joint_to_text/configs/mustc_noise.list \
        --data-path ${must_c_en_de_src_text} \
        --out-path ${must_c_en_de_src_text_pho}
```
-   Replace the source text under the "src_text" column in the tsv file with the corresponding phoneme reprentation generated in the step above.
Below is the snapshot for the MuST-C en-de dev tsv
```
id  audio   n_frames    tgt_text    src_text    speaker
ted_767_0   en-de/flac.zip:10071514743:48445    56160   Heute spreche ich zu Ihnen über Energie und Klima.  ▁AY1 M ▁G OW1 IH0 NG ▁T UW1 ▁T AO1 K ▁T AH0 D EY1 ▁AH0 B AW1 T ▁EH1 N ER0 JH IY0 ▁AH0 N D ▁K L AY1 M AH0 T  spk.767_
ted_767_1   en-de/flac.zip:1214217978:205678    226080  Und das überrascht vielleicht etwas, weil sich meine Vollzeitbeschäftigung bei der Stiftung hauptsächlich um Impfstoffe und Saatgut dreht, um die Dinge, die wir erfinden und liefern müssen um den ärmsten 2 Milliarden ein besseres Leben zu ermöglichen. ▁AH0 N D ▁DH AE1 T ▁M AY1 T ▁S IY1 M ▁AH0 ▁B IH1 T ▁S ER0 P R AY1 Z IH0 NG ▁B IH0 K AO1 Z ▁M AY1 ▁F UH1 L ▁T AY1 M ▁W ER1 K ▁AE1 T ▁DH AH0 ▁F AW0 N D EY1 SH AH0 N ▁IH1 Z ▁M OW1 S T L IY0 ▁AH0 B AW1 T ▁V AE2 K S IY1 N Z ▁AH0 N D ▁S IY1 D Z ▁AH0 B AW1 T ▁DH AH0 ▁TH IH1 NG Z ▁DH AE1 T ▁W IY1 ▁N IY1 D ▁T UW1 ▁IH0 N V EH1 N T ▁AH0 N D ▁D IH0 L IH1 V ER0 ▁T UW1 ▁HH EH1 L P ▁DH AH0 ▁P UH1 R IH0 S T ▁T UW1 ▁B IH1 L Y AH0 N ▁L AY1 V ▁B EH1 T ER0 ▁L IH1 V Z spk.767_
```
-   Prepare phoneme dictionary and save to $MANIFEST_ROOT as [src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/src_dict.txt)
#### Prepare WMT text data
-   [Download wmt data](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh)
-   Convert source text (English) into phoneme representation as above
-   Generate binary parallel files with "fairseq-preprocess" from fairseq for training and validation. The source input is English phoneme representation and the target input is German sentencepiece token .  The output is saved under $parallel_text_data

## Training
The model is trained with 8 v100 GPUs.

#### Download pretrained models
-    [pretrain_encoder](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)
-    [pretrain_nmt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_mt.pt)

#### Training scripts
- Jointly trained model from scratch
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_s \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.001 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000  \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '"'"'▁NOISE'"'"' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml config.yaml \
    --keep-last-epochs 10
```
- Jointly trained model with good initialization, cross attentive loss and online knowledge distillation
```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --num-workers 8 \
    --task speech_text_joint_to_text \
    --arch dualinputs2ttransformer_m \
    --user-dir examples/speech_text_joint_to_text \
    --max-epoch 100 --update-mix-data \
    --optimizer adam --lr-scheduler inverse_sqrt \
    --lr 0.002 --update-freq 4 --clip-norm 10.0 \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
    --label-smoothing 0.1 --max-tokens 10000 --max-tokens-text 10000 \
    --max-positions-text 400 --seed 2 --speech-encoder-layers 12 \
    --text-encoder-layers 6 --encoder-shared-layers 6 --decoder-layers 6 \
    --dropout 0.1 --warmup-updates 20000 --attentive-cost-regularization 0.02 \
    --text-sample-ratio 0.25 --parallel-text-data ${parallel_text_data} \
    --text-input-cost-ratio 0.5 --enc-grad-mult 2.0 --add-speech-eos \
    --log-format json --langpairs en-de --noise-token '"'"'▁NOISE'"'"' \
    --mask-text-ratio 0.0 --max-tokens-valid 20000 --ddp-backend no_c10d \
    --log-interval 100 --data-buffer-size 50 --config-yaml config.yaml \
    --load-pretrain-speech-encoder ${pretrain_encoder} \
    --load-pretrain-decoder ${pretrain_nmt} \
    --load-pretrain-text-encoder-last ${pretrain_nmt} \
    --keep-last-epochs 10
```

## Evaluation
```bash
python ./fairseq_cli/generate.py \
        ${MANIFEST_ROOT} \
        --task speech_text_joint_to_text \
        --max-tokens 25000 \
        --nbest 1 \
        --results-path ${infer_results} \
        --batch-size 512 \
        --path ${model} \
        --gen-subset tst-COMMON_st \
        --config-yaml config.yaml \
        --scoring sacrebleu \
        --beam 5 --lenpen 1.0 \
        --user-dir examples/speech_text_joint_to_text \
        --load-speech-only
```

## Results (Joint training with initialization + CAR + online KD)
|Direction|En-De | En-Es | En-Fr |
|---|---|---|---|
|BLEU|27.4| 31.2 | 37.6 |
|checkpoint | [link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/checkpoint_ave_10.pt) |[link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_es/checkpoint_ave_10.pt)|[link](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_fr/checkpoint_ave_10.pt)|
