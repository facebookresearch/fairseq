TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=5000      
LR=0.0
MAX_TOKENS=2048
UPDATE_FREQ=1
LABEL_SMOOTHING=0.0
MAX_UPDATE=20000
MAX_EPOCH=1
BART_PATH=/fsx/akshats/checkpoints/nlu/stop/bart_full_granular_stop/bart_base_save/bart_base_save.bart_base.ls0.0.mt2048.uf1.me50.bsz32.dr0.2.atdr0.1.actdr0.0.wd0.adam.beta9999.eps1e-10.clip0.0.lr2.04e-05.warm5000.ngpu8/checkpoint50.pt
CUDA_VISIBLE_DEVICES=0 fairseq-train /fsx/akshats/data/stop_updated_granular_everything/bart_asr_eval/bart-base-bin \
    --user-dir examples/audio_nlp/nlu/src \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task nlu_translation \
    --source-lang utterance --target-lang parse_decoupled_bart \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing $LABEL_SMOOTHING \
    --dropout 0.2 --attention-dropout 0.1 --relu-dropout 0.0 \
    --weight-decay 0 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-10 \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr $LR --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --max-update $MAX_UPDATE \
    --max-epoch $MAX_EPOCH \
    --required-batch-size-multiple 1 \
    --skip-invalid-size-inputs-valid-test \
    --valid-subset test_weather \
    --eval-bleu-args '{"beam": 1}' \
    --train-subset eval_no \
    --find-unused-parameters;