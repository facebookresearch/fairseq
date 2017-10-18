TEXT=/home/playma/4t/playma/Experiment/fairseq-py/data
OUT=/home/playma/4t/playma/Experiment/fairseq-py/data-bin 

python3 preprocess.py --source-lang source --target-lang target \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir $OUT \
  --nwordstgt 4000 \
  --nwordssrc 4000

