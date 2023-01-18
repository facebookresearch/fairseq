
TASK_DIR=/fsx/akshats/data/stop_updated_granular/full
MODEL_DIR=/fsx/akshats/models/bart.base

fairseq-preprocess \
  --source-lang "utterance" \
  --target-lang "parse_decoupled_bart" \
  --trainpref "${TASK_DIR}/train.bpe" \
  --validpref "${TASK_DIR}/eval.bpe" \
  --testpref "${TASK_DIR}/test.bpe" \
  --destdir "${TASK_DIR}-bart-base-bin/" \
  --workers 60 \
  --srcdict "$MODEL_DIR/dict.txt" \
  --tgtdict "$MODEL_DIR/dict.txt";