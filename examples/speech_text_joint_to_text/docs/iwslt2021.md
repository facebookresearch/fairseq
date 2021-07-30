[[Back]](..)

# Joint Speech Text Training for the 2021 IWSLT multilingual speech translation

This directory contains the code from paper ["FST: the FAIR Speech Translation System for the IWSLT21 Multilingual Shared Task"](https://arxiv.org/pdf/2107.06959.pdf).

## Prepare Data
#### Download files
-   Sentence piece model [spm.model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/spm.model)
-   Dictionary [tgt_dict.txt](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/dict.txt)
-   Config [config.yaml](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/config.yaml)

#### Prepare
-   [Please follow the data preparation in speech-to-text](https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/docs/mtedx_example.md)



## Training

#### Download pretrained models
- [Pretrained mbart model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/mbart.pt)
- [Pretrained w2v model](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/xlsr_53_56k.pt)


#### Training scripts

```bash
python train.py ${MANIFEST_ROOT} \
    --save-dir ${save_dir} \
    --user-dir examples/speech_text_joint_to_text \
    --train-subset train_es_en_tedx,train_es_es_tedx,train_fr_en_tedx,train_fr_es_tedx,train_fr_fr_tedx,train_it_it_tedx,train_pt_en_tedx,train_pt_pt_tedx \
    --valid-subset valid_es_en_tedx,valid_es_es_tedx,valid_es_fr_tedx,valid_es_it_tedx,valid_es_pt_tedx,valid_fr_en_tedx,valid_fr_es_tedx,valid_fr_fr_tedx,valid_fr_pt_tedx,valid_it_en_tedx,valid_it_es_tedx,valid_it_it_tedx,valid_pt_en_tedx,valid_pt_es_tedx,valid_pt_pt_tedx \
    --config-yaml config.yaml --ddp-backend no_c10d \
    --num-workers 2 --task speech_text_joint_to_text \
    --criterion guided_label_smoothed_cross_entropy_with_accuracy \
    --label-smoothing 0.3 --guide-alpha 0.8 \
    --disable-text-guide-update-num 5000 --arch dualinputxmtransformer_base \
    --max-tokens 500000 --max-sentences 3 --max-tokens-valid 800000 \
    --max-source-positions 800000 --enc-grad-mult 2.0 \
    --attentive-cost-regularization 0.02 --optimizer adam \
    --clip-norm 1.0 --log-format simple --log-interval 200 \
    --keep-last-epochs 5 --seed 1 \
    --w2v-path ${w2v_path} \
    --load-pretrained-mbart-from ${mbart_path} \
    --max-update 1000000 --update-freq 4 \
    --skip-invalid-size-inputs-valid-test \
    --skip-encoder-projection --save-interval 1 \
    --attention-dropout 0.3 --mbart-dropout 0.3 \
    --finetune-w2v-params all --finetune-mbart-decoder-params all \
    --finetune-mbart-encoder-params all --stack-w2v-mbart-encoder \
    --drop-w2v-layers 12 --normalize \
    --lr 5e-05 --lr-scheduler inverse_sqrt --warmup-updates 5000
```

## Evaluation
```bash
python ./fairseq_cli/generate.py
   ${MANIFEST_ROOT} \
   --task speech_text_joint_to_text \
   --user-dir ./examples/speech_text_joint_to_text \
   --load-speech-only  --gen-subset  test_es_en_tedx \
   --path  ${model}  \
   --max-source-positions 800000 \
   --skip-invalid-size-inputs-valid-test \
   --config-yaml config.yaml \
   --infer-target-lang en  \
   --max-tokens 800000 \
   --beam 5 \
   --results-path ${RESULTS_DIR}  \
   --scoring sacrebleu
```
The trained model can be downloaded [here](https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/checkpoint17.pt)

|direction|es_en|fr_en|pt_en|it_en|fr_es|pt_es|it_es|es_es|fr_fr|pt_pt|it_it|
|---|---|---|---|---|---|---|---|---|---|---|---|
|BLEU|31.62|36.93|35.07|27.12|38.87|35.57|34.13|74.59|74.64|70.84|69.76|
