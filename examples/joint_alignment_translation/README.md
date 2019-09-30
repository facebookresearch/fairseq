# Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)

This page includes instructions for training models described in [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](https://arxiv.org/abs/1909.02074).

## Training a joint alignment-translation model on WMT'18 En-De

##### 1. Extract and preprocess the WMT'18 En-De data
```bash
./prepare-wmt18en2de_no_norm_no_escape_no_agressive.sh
```

##### 2. Generate alignments from statistical alignment toolkits e.g. Giza++/FastAlign.
In this example, we use FastAlign.
```bash
git clone git@github.com:clab/fast_align.git
pushd fast_align
mkdir build
cd build
cmake ..
make
popd
ALIGN=fast_align/build/fast_align
paste bpe.32k/train.en bpe.32k/train.de | awk -F '\t' '{print $1 " ||| " $2}' > bpe.32k/train.en-de
$ALIGN -i bpe.32k/train.en-de -d -o -v > bpe.32k/train.align
```

##### 3. Preprocess the dataset with the above generated alignments.
```bash
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref bpe.32k/train \
    --validpref bpe.32k/valid \
    --testpref bpe.32k/test \
    --align-suffix align \
    --destdir binarized/ \
    --joined-dictionary \
    --workers 32
```

##### 4. Train a model
```bash
fairseq-train \
    binarized \
    --arch transformer_wmt_en_de_big_align --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --activation-fn relu\
    --lr 0.0002 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 3500 --label-smoothing 0.1 \
    --save-dir ./checkpoints --log-interval 1000 --max-update 60000 \
    --keep-interval-updates -1 --save-interval-updates 0 \
    --load-alignments --criterion label_smoothed_cross_entropy_with_alignment \
    --fp16
```

Note that the `--fp16` flag requires you have CUDA 9.1 or greater and a Volta GPU or newer.

If you want to train the above model with big batches (assuming your machine has 8 GPUs):
- add `--update-freq 8` to simulate training on 8x8=64 GPUs
- increase the learning rate; 0.0007 works well for big batches

##### 5. Evaluate and generate the alignments (BPE level)
```bash
fairseq-generate \
    binarized --gen-subset test --print-alignment \
    --source-lang en --target-lang de \
    --path checkpoints/checkpoint_best.pt --beam 5 --nbest 1
```

##### 6. Other resources.
The code for:
1. preparing alignment test sets
2. converting BPE level alignments to token level alignments
3. symmetrizing bidirectional alignments
4. evaluating alignments using AER metric
can be found [here](https://github.com/lilt/alignment-scripts)

## Citation

```bibtex
@inproceedings{garg2019jointly,
  title = {Jointly Learning to Align and Translate with Transformer Models},
  author = {Garg, Sarthak and Peitz, Stephan and Nallasamy, Udhyakumar and Paulik, Matthias},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  address = {Hong Kong},
  month = {November},
  url = {https://arxiv.org/abs/1909.02074},
  year = {2019},
}
```
