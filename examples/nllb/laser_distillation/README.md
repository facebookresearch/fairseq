
# LASER3 Training Code


LASER3 encoders are built from LASER encoders following a [teacher-student approach](https://arxiv.org/abs/2205.12654).
This folder contains the training code for LASER3 encoders.

If you are interested in the training code of LASER encoders (the Teacher), you can refer to the [LASER's training code](../../laser/README.md). To learn more about LASER, you can refer to the [LASER repository](https://github.com/facebookresearch/LASER/). If you are interested in inference only, you can refer to [this page](https://github.com/facebookresearch/LASER/tree/main/nllb/README.md).


# Training the language specialized encoders

The data preparation is similar to LASER's [training code](../../laser/README.md).

Binarize your data with fairseq, as described [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#data-pre-processing), by using the argument `--dataset-impl cached`.

Then, create a json config file with this format:
```
{
  "src_vocab": "/path/to/spm.src.cvocab",
  "tgt_vocab": "/path/to/spm.tgt.cvocab",
  "train": [
    {
      "id": 0,
      "src": "/path/to/srclang1-tgtlang0/train.srclang1",
      "tgt": "/path/to/srclang1-tgtlang0/train.tgtlang0"
    },
    {
      "id": 0,
      "src": "/path/to/srclang1-tgtlang1/train.srclang1",
      "tgt": "/path/to/srclang1-tgtlang1/train.tgtlang1"
    },
    {
      "id": 0,
      "src": "/path/to/srclang2-tgtlang0/train.srclang2",
      "tgt": "/path/to/srclang2-tgtlang0/train.tgtlang0"
    },
    {
      "id": 0,
      "src": "/path/to/srclang2-tgtlang1/train.srclang2",
      "tgt": "/path/to/srclang2-tgtlang1/train.tgtlang1"
    },
    {
      "id": 0,
      "src": "/path/to/srclang3-srclang3/train.srclang3",
      "tgt": "/path/to/srclang3-srclang3/train.srclang3"
    },
    ...
  ]
}
```
where paths are paths to binarized indexed fairseq dataset files. Please note that binarized file paths don't contain the `.idx` and `.bin` postfixes. `id` represents the target language id. This is required for regular LASER training but can be set to zero for teacher-student training.

`src_vocab` and `src` correspond to the student, whereas `tgt_vocab` and `tgt` correspond to the teacher.

The section `/path/to/srclang3-srclang3/train.srclang3` with same arguments for both `src` and `tgt` corresponds to monolingual data that can be used for Masked Language Model training.


# Training Command Line Example

```bash
fairseq-train \
    /path/to/configfile.json   # the configuration file described above \
    --user-dir examples/nllb/laser_distillation \
    --log-interval 100  \
    --log-format simple \
    --task laser_distillation \
    --arch laser_transformer \
    --sentemb-criterion cls \
    --prepend-bos  \
    --criterion encoder_similarity \
    --save-dir /path/to/savedir       # where the checkpoints will be saved \
    --optimizer adam  \
    --adam-betas '(0.9,0.98)'  \
    --adam-eps 1e-6 \
    --lr 0.0001 \
    --warmup-updates 1000 \
    --clip-norm 5 \
    --update-freq 1 \
    --dropout 0.3 \
    --max-tokens 2000 \
    --max-epoch 50 \
    --left-pad-source False \
    --left-pad-target True \
    --encoder-embed-dim 1024 \
    --encoder-layers 12 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 4 \
    --decoder-embed-dim 1 \
    --decoder-layers 1 \
    --decoder-embed-dim 1 \
    --decoder-ffn-embed-dim 1 \
    --decoder-attention-heads 1 \
    --decoder-lang-embed-dim 1 \
    --ddp-backend=no_c10d \
    --save-interval-updates 30000 \
    --disable-validation \
    --teacher-checkpoint-path /path/to/teacherencoder.py  # For example path to Laser2 if you want to use Laser2 as teacher \
    --lambda-self 1.0 \
    --lambda-mask 0.09 \
    --lambda-distil 1.0 \
    --student-teacher-config "mask:lang1-lang1,self:lang2-lang2,distil:lang3-lang4" \
    --joint-weighting-alpha 0.09

```

The `laser_distillation` task can actually perform 3 tasks: mask, distil, self.
The `--student-teacher-config` parameter describes how you want to combine these. The parameter accepts a series of tasks separated by commas.

- `mask`: a Masked Language Model training is performed using monolingual data.
- `distil`: a distillation task, the Student encoder is trained with cosine loss based on the Teacher's sentence embedding output.
- `self`: same as `distil` but the same samples are sent to the Teacher and the Student encoders.
