
/Users/csinclair/opt/anaconda3/envs/fairseq/bin/fairseq-interactive  \
  --path model_1/:model_2/:model_3/ \
  --task translation_marian \
  --beam 5

  fairseq-interactive \
  --path 


/Users/csinclair/opt/anaconda3/envs/fairseq/bin/fairseq-interactive  wmt19.en-ru.ensemble \
--task translation \
--path wmt19.en-ru.ensemble/model1.pt \
--source-lang en \
--target-lang ru \
--beam 5 

fairseq-train /Users/csinclair/src/translation-models-data/data/datasets/processed/fairseq/mbart/baseline/bin \
--decoder-normalize-before --maximize-best-checkpoint-metric \
--log-interval 10 --adam-betas '(0.9, 0.98)' --save-interval 1000 --lr 3e-05 --warmup-updates 2500 \
--save-interval-updates 1000  --weight-decay 0.0 --keep-best-checkpoints 1 \
--validate-interval 10000 --lr-scheduler inverse_sqrt --max-source-positions 1024 \
--eval-bleu-detok 'sentencepiece' \
--eval-bleu-detok-args '{"sentencepiece_model": "/Users/csinclair/Downloads/mbart50.ft.1n/sentence.bpe.model"}' \
--eval-bleu-remove-bpe 'sentencepiece' \
--task translation_from_pretrained_multi_bart --adam-eps 1e-06 --dropout 0.3 \
--encoder-normalize-before --no-save-optimizer-state --log-format simple \
--max-target-positions 1024 --best-checkpoint-metric bleu \
--max-sentences 1 \
--criterion label_smoothed_cross_entropy --seed 42 \
--finetune-from-model /Users/csinclair/Downloads/mbart50.ft.1n/model.pt \
--attention-dropout 0.1 \
--optimizer adam --layernorm-embedding \
--source-lang en_XX --target-lang es_XX \
--cpu \
--patience 5  --keep-interval-updates 1 \
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI \
--max-tokens 1024 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --scoring sacrebleu \
--max-update 0 --validate-interval-updates 1 --max-tokens-valid 1024 \
--eval-bleu  --label-smoothing 0.2 --update-freq 1 --arch mbart_large \
--no-last-checkpoints --eval-bleu-print-samples 




>>>>>>> f003703bf0ce7f884ec1c359a5741c270ebe044a
fairseq-train /Users/csinclair/src/translation-models-data/data/datasets/processed/fairseq/mbart/baseline/bin \
--decoder-normalize-before --maximize-best-checkpoint-metric \
--log-interval 10 --adam-betas '(0.9, 0.98)' --save-interval 1000 --lr 3e-05 --warmup-updates 2500 \
--save-interval-updates 1000  --weight-decay 0.0 --keep-best-checkpoints 1 \
--validate-interval 10000 --lr-scheduler inverse_sqrt --max-source-positions 1024 \
--task translation_multi_simple_epoch --adam-eps 1e-06 --dropout 0.3 \
--encoder-normalize-before --no-save-optimizer-state --log-format simple \
--max-target-positions 1024 --best-checkpoint-metric bleu \
--max-sentences 5 \
--criterion label_smoothed_cross_entropy --seed 42 \
--finetune-from-model /Users/csinclair/Downloads/mbart50.ft.1n/model.pt \
--attention-dropout 0.1 \
--optimizer adam --layernorm-embedding \
--patience 5 \
--cpu \
--sampling-method "temperature" \
--sampling-temperature "1.5" \
--encoder-langtok "src" \
--decoder-langtok \
--patience 5  --keep-interval-updates 1 \
--lang-dict /Users/csinclair/Downloads/mbart50.ft.1n/ML50_langs.txt \
--lang-pairs 'en_XX-af_ZA,en_XX-ar_AR,en_XX-az_AZ,en_XX-bn_IN,en_XX-cs_CZ,en_XX-de_DE,en_XX-es_XX,en_XX-et_EE,en_XX-fa_IR,en_XX-fi_FI,en_XX-fr_XX,en_XX-gl_ES,en_XX-gu_IN,en_XX-he_IL,en_XX-hi_IN,en_XX-hr_HR,en_XX-id_ID,en_XX-it_IT,en_XX-ja_XX,en_XX-ka_GE,en_XX-kk_KZ,en_XX-km_KH,en_XX-ko_KR,en_XX-lt_LT,en_XX-lv_LV,en_XX-mk_MK,en_XX-ml_IN,en_XX-mn_MN,en_XX-mr_IN,en_XX-my_MM,en_XX-ne_NP,en_XX-nl_XX,en_XX-pl_PL,en_XX-ps_AF,en_XX-pt_XX,en_XX-ro_RO,en_XX-ru_RU,en_XX-si_LK,en_XX-sl_SI,en_XX-sv_SE,en_XX-ta_IN,en_XX-te_IN,en_XX-th_TH,en_XX-tr_TR,en_XX-uk_UA,en_XX-ur_PK,en_XX-vi_VN,en_XX-xh_ZA,en_XX-zh_CN' \
--max-tokens 1024 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --scoring sacrebleu \
--max-update 0 --validate-interval-updates 1 --max-tokens-valid 1024 \
--label-smoothing 0.2 --update-freq 1 --arch mbart_large \
--no-last-checkpoints --eval-bleu-print-samples   \
--eval-bleu \
--eval-bleu-detok 'sentencepiece' \
--eval-bleu-detok-args '{"sentencepiece_model": "/Users/csinclair/Downloads/mbart50.ft.1n/sentence.bpe.model"}' \
--eval-bleu-remove-bpe 'sentencepiece' \





fairseq-interactive /Users/csinclair/src/translation-models-data/data/datasets/processed/fairseq/mbart/baseline/bin \
--path /Users/csinclair/Downloads/mbart50.ft.1n/model.pt \
  --task translation_multi_simple_epoch \
  --target-lang es_XX --source-lang en_XX \
  --bpe 'sentencepiece' --sentencepiece-model /Users/csinclair/Downloads/mbart50.ft.1n/sentence.bpe.model \
   --remove-bpe 'sentencepiece' \
   --encoder-langtok "src" \
  --decoder-langtok \
  --lang-pairs "en_XX-es_XX" \
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI 




fairseq-generate processed \
  --path model/mbart50.ft.1n/model.pt \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang en_XX \
  --target-lang es_XX \
  --sacrebleu --remove-bpe 'sentencepiece'\
  --batch-size 14 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict model/mbart50.ft.1n/ML50_langs.txt \
 --lang-pairs 'en_XX-af_ZA,en_XX-ar_AR,en_XX-az_AZ,en_XX-bn_IN,en_XX-cs_CZ,en_XX-de_DE,en_XX-es_XX,en_XX-et_EE,en_XX-fa_IR,en_XX-fi_FI,en_XX-fr_XX,en_XX-gl_ES,en_XX-gu_IN,en_XX-he_IL,en_XX-hi_IN,en_XX-hr_HR,en_XX-id_ID,en_XX-it_IT,en_XX-ja_XX,en_XX-ka_GE,en_XX-kk_KZ,en_XX-km_KH,en_XX-ko_KR,en_XX-lt_LT,en_XX-lv_LV,en_XX-mk_MK,en_XX-ml_IN,en_XX-mn_MN,en_XX-mr_IN,en_XX-my_MM,en_XX-ne_NP,en_XX-nl_XX,en_XX-pl_PL,en_XX-ps_AF,en_XX-pt_XX,en_XX-ro_RO,en_XX-ru_RU,en_XX-si_LK,en_XX-sl_SI,en_XX-sv_SE,en_XX-ta_IN,en_XX-te_IN,en_XX-th_TH,en_XX-tr_TR,en_XX-uk_UA,en_XX-ur_PK,en_XX-vi_VN,en_XX-xh_ZA,en_XX-zh_CN' > roblox_en_XX-es_XX.tgt &&


fairseq-generate test_set \
  --path model/mbart50.ft.1n/model.pt \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang en_XX \
  --target-lang es_XX \
  --sacrebleu --remove-bpe 'sentencepiece'\
  --batch-size 14 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict model/mbart50.ft.1n/ML50_langs.txt \
 --lang-pairs 'en_XX-af_ZA,en_XX-ar_AR,en_XX-az_AZ,en_XX-bn_IN,en_XX-cs_CZ,en_XX-de_DE,en_XX-es_XX,en_XX-et_EE,en_XX-fa_IR,en_XX-fi_FI,en_XX-fr_XX,en_XX-gl_ES,en_XX-gu_IN,en_XX-he_IL,en_XX-hi_IN,en_XX-hr_HR,en_XX-id_ID,en_XX-it_IT,en_XX-ja_XX,en_XX-ka_GE,en_XX-kk_KZ,en_XX-km_KH,en_XX-ko_KR,en_XX-lt_LT,en_XX-lv_LV,en_XX-mk_MK,en_XX-ml_IN,en_XX-mn_MN,en_XX-mr_IN,en_XX-my_MM,en_XX-ne_NP,en_XX-nl_XX,en_XX-pl_PL,en_XX-ps_AF,en_XX-pt_XX,en_XX-ro_RO,en_XX-ru_RU,en_XX-si_LK,en_XX-sl_SI,en_XX-sv_SE,en_XX-ta_IN,en_XX-te_IN,en_XX-th_TH,en_XX-tr_TR,en_XX-uk_UA,en_XX-ur_PK,en_XX-vi_VN,en_XX-xh_ZA,en_XX-zh_CN' > generic_en_XX-es_XX.tgt

fairseq-interactive wmt19.en-ru.ensemble \
--task translation \
--path wmt19.en-ru.ensemble/model_test.pt \
--source-lang en \
--target-lang ru \
--beam 5 
>>>>>>> Stashed changes
=======
>>>>>>> f003703bf0ce7f884ec1c359a5741c270ebe044a
