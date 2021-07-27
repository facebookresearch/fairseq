fairseq-generate $path_2_data \
  --path $model \
  --task translation_marian \
  --gen-subset test \
  --source-lang en \
  --target-lang de \
  --sacrebleu --remove-bpe 'sentencepiece'\
  --batch-size 32 \