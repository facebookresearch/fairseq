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
-   [Please follow the data preparation in the S2T example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md)
-   Append src_text in the tsv file with phoneme representation.
```bash
    python examples/speech_text_joint_to_text/scripts/g2p_encode.py \
        --lower-case --do-filter --use-word-start --no-punc \
        --reserve-word examples/speech_text_joint_to_text/configs/mustc_noise.list \
        --data-path ${must_c_en_de_src_text} \
        --out-path ${must_c_en_de_src_text_pho}
```
-   Update tsv data with src_text generated above and save to $MANIFEST_ROOT
-   Prepare phoneme dictionary and save to $MANIFEST_ROOT as [src_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/must_c/en_de/src_dict.txt)
#### Prepare WMT text data
-   [Download wmt data](https://github.com/pytorch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh)
-   Convert source text (English) into phoneme representation as above
-   Generate binary parallel file for training (as translation example) and save data in $parallel_text_data

## Training
The model is trained with 8 v100 GPUs.

#### Download pretrained models
-    [pretrain_encoder](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_multilingual_asr_transformer_m.pt)
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
        --gen-subset tst-COMMON \
        --config-yaml config_spm.yaml \
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
