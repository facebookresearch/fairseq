fairseq-train /Users/csinclair/src/translation-models-data/data/datasets/processed/fairseq/mbart/baseline/bin \
--decoder-normalize-before --maximize-best-checkpoint-metric \
--log-interval 10 --adam-betas '(0.9, 0.98)' --save-interval 1000 --lr 3e-05 --warmup-updates 2500 \
--save-interval-updates 1000  --weight-decay 0.0 --keep-best-checkpoints 1 \
--validate-interval 10000 --lr-scheduler inverse_sqrt --max-source-positions 1024 \
--eval-bleu-detok 'sentencepiece' \
--eval-bleu-detok-args '{"sentencepiece_model": "/Users/csinclair/Downloads/mbart50.ft.1n/sentence.bpe.model"}' \
--eval-bleu-remove-bpe 'sentencepiece' \
--task translation_from_pretrained_bart --adam-eps 1e-06 --dropout 0.3 \
--encoder-normalize-before --no-save-optimizer-state --log-format simple \
--max-target-positions 1024 --best-checkpoint-metric bleu \
--max-sentences 1 \
--criterion label_smoothed_cross_entropy --seed 42 \
--finetune-from-model /Users/csinclair/Downloads/mbart50.ft.1n/model.pt \
--attention-dropout 0.1 --source-lang en_XX --target-lang es_XX \
--optimizer adam --layernorm-embedding \
--cpu \
--patience 5  --keep-interval-updates 1 \
--langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI \
--max-tokens 1024 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --scoring sacrebleu \
--max-update 0 --validate-interval-updates 1 --max-tokens-valid 1024 \
--eval-bleu  --label-smoothing 0.2 --update-freq 1 --arch mbart_large \
--no-last-checkpoints --eval-bleu-print-samples 




fairseq-train /Users/csinclair/src/translation-models-data/data/datasets/processed/fairseq/mbart/baseline/bin \
--decoder-normalize-before --maximize-best-checkpoint-metric \
--log-interval 10 --adam-betas '(0.9, 0.98)' --save-interval 1000 --lr 3e-05 --warmup-updates 2500 \
--save-interval-updates 1000  --weight-decay 0.0 --keep-best-checkpoints 1 \
--validate-interval 10000 --lr-scheduler inverse_sqrt --max-source-positions 1024 \
--eval-bleu-detok 'sentencepiece' \
--eval-bleu-detok-args '{"sentencepiece_model": "/Users/csinclair/Downloads/mbart50.ft.1n/sentence.bpe.model"}' \
--eval-bleu-remove-bpe 'sentencepiece' \
--task translation_multi_simple_epoch --adam-eps 1e-06 --dropout 0.3 \
--encoder-normalize-before --no-save-optimizer-state --log-format simple \
--max-target-positions 1024 --best-checkpoint-metric bleu \
--max-sentences 1 \
--criterion label_smoothed_cross_entropy --seed 42 \
--finetune-from-model /Users/csinclair/Downloads/mbart50.ft.1n/model.pt \
--attention-dropout 0.1 --source-lang en_XX --target-lang es_XX \
--optimizer adam --layernorm-embedding \
--patience 5 \
--cpu \
--sampling-method "temperature" \
--sampling-temperature "1.5" \
--encoder-langtok "src" \
--decoder-langtok \
--patience 5  --keep-interval-updates 1 \
--lang-dict /Users/csinclair/Downloads/mbart50.ft.1n/ML50_langs.txt \
--lang-pairs 'en_XX-es_XX' \
--max-tokens 1024 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --scoring sacrebleu \
--max-update 0 --validate-interval-updates 1 --max-tokens-valid 1024 \
--eval-bleu  --label-smoothing 0.2 --update-freq 1 --arch mbart_large \
--no-last-checkpoints --eval-bleu-print-samples 




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
