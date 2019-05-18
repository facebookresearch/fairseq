Example usage:
```
# Download dataset, and create a modified dataset; download pretrained word vectors
# NOTE: The code base is in the fairseq directory, so all paths are relative to fairseq
$ cd examples/transaction_bot/
# Download dataset manually as follows:
# (1) Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
# (2) Download dialog-bAbI-tasks_1_.tgz in directory fairseq/examples/transaction_bot
$ tar zxvf dialog-bAbI-tasks_1_.tgz
$ python3 create-fairseq-dialog-dataset.py

$ mkdir pretrained-word-vectors
$ cd pretrained-word-vectors
$ Go to https://nlp.stanford.edu/projects/glove/ and download glove.6B.zip; it will take some time to download
$ unzip glove.6B.zip
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/transaction_bot/fairseq-dialog-dataset/task1
$ python3 -m pdb preprocess.py --task transaction_bot --source-lang hmn --target-lang bot --joined-dictionary --trainpref $TEXT/task1-trn --validpref $TEXT/task1-dev --testpref $TEXT/task1-tst --destdir data-bin/transaction_bot/task1

# Train the model (better for a single GPU setup):
# ***NOTE*** if Training must be started from the beginning then the checkpoint files
# in the directory "checkpoints/transaction_bot/task1" must be removed, otherwise training
# will resume from the last best checkpoint model
$ rm -r checkpoints
$ mkdir -p checkpoints/transaction_bot/task1 (remove this line; not needed)
$ CUDA_VISIBLE_DEVICES=0 python3 -m pdb train.py --task transaction_bot data-bin/transaction_bot/task1 --arch lstm_transaction_bot_1 --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --save-dir checkpoints/transaction_bot/task1

# Generate:
$ python3 -m pdb generate.py --task transaction_bot data-bin/transaction_bot/task1 --path checkpoints/transaction_bot/task1/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe

```
