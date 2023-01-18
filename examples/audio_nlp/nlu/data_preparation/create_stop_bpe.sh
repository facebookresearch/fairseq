# process stop seq2seq fairseq data with BPE
TASK_DIR=/fsx/akshats/data/stop_asr_hypo_granular
MODEL_DIR=/fsx/akshats/models/bart.base

wget -P "$TASK_DIR/" -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -P "$TASK_DIR/" -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -P "$TASK_DIR/" -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

for SPLIT in train eval test
do
  for LANG in librispeech_asr_hypo_utterance parse_decoupled_bart
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json "$TASK_DIR/encoder.json" \
    --vocab-bpe "$TASK_DIR/vocab.bpe" \
    --inputs "$TASK_DIR/$SPLIT.$LANG" \
    --outputs "$TASK_DIR/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "librispeech_asr_hypo_utterance" \
  --target-lang "parse_decoupled_bart" \
  --trainpref "${TASK_DIR}/train.bpe" \
  --validpref "${TASK_DIR}/eval.bpe" \
  --testpref "${TASK_DIR}/test.bpe" \
  --destdir "${TASK_DIR}/libri-bart-base-bin/" \
  --workers 60 \
  --srcdict "$MODEL_DIR/dict.txt" \
  --tgtdict "$MODEL_DIR/dict.txt";