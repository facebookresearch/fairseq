FAIR Sequence-to-Sequence Toolkit for Story Generation

The following commands provide an example of pre-processing data, training a model, and generating text for story generation with the WritingPrompts dataset. 

The dataset can be downloaded like this:

```
curl https://s3.amazonaws.com/fairseq-py/data/writingPrompts.tar.gz | tar xvjf -
```

and contains a train, test, and valid split. The dataset is described here: https://arxiv.org/abs/1805.04833, where only the first 1000 words of each story are modeled. 


Example usage:
```
# Binarize the dataset:
$ TEXT=examples/stories/writingPrompts
$ python preprocess.py --source-lang wp_source --target-lang wp_target \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/writingPrompts --thresholdtgt 10 --thresholdsrc 10

# Train the model:
$ python train.py data-bin/writingPrompts -a fconv_self_att_wp --lr 0.25 --clip-norm 0.1 --max-tokens 1500 --lr-scheduler reduce_lr_on_plateau --decoder-attention True --encoder-attention False --criterion label_smoothed_cross_entropy --weight-decay .0000001 --label-smoothing 0 --source-lang wp_source --target-lang wp_target --gated-attention True --self-attention True --project-input True --pretrained False

# Train a fusion model:
# add the arguments: --pretrained True --pretrained-checkpoint path/to/checkpoint

# Generate:
$ python generate.py data-bin/writingPrompts --path /path/to/trained/model/checkpoint_best.pt --batch-size 32 --beam 1 --sampling --sampling-topk 10 --sampling-temperature 0.8 --nbest 1 
```
