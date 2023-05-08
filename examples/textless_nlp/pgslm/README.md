# Text-Free Prosody-Aware Generative Spoken Language Modeling

This folder contains code and recipes to reproduce results reported in a paper _Text-Free Prosody-Aware Generative Spoken Language Modeling_,
Eugene Kharitonov*, Ann Lee*, Adam Polyak, Yossi Adi, Jade Copet, Kushal Lakhotia, Tu-Anh Nguyen, Morgane Rivière, Abdelrahman Mohamed, Emmanuel Dupoux, Wei-Ning Hsu, 2021. arxiv/2109.03264 [[arxiv]](https://arxiv.org/abs/2109.03264).

`*` denotes equal contribution.

You can find demo samples [[here]](https://speechbot.github.io/pgslm/index.html).

<details>
  <summary>If you find this code useful, please consider citing our work using this bibtex </summary>
  
```
  @misc{Kharitonov2021,
      title={Text-Free Prosody-Aware Generative Spoken Language Modeling}, 
      author={Eugene Kharitonov and Ann Lee and Adam Polyak and Yossi Adi and Jade Copet and Kushal Lakhotia and Tu-Anh Nguyen and Morgane Rivière and Abdelrahman Mohamed and Emmanuel Dupoux and Wei-Ning Hsu},
      year={2021},
      eprint={2109.03264},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
</details>


## Additional requirements
Three packages are required in addition to fairseq, they are installable with pip:
```bash 
pip install AMFM-decompy SoundFile scipy sklearn torchaudio npy-append-array
```

## Data preprocessing

### Prepare unit pseudo-text transcriptions of the audio
To get unit trascripts of the speech data we rely on the preprocessing steps of [GSLM](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit/) work.

Firstly, we will need to prepare manifest files for the dataset we want to preprocess
```
mkdir manifests/
python examples/wav2vec/wav2vec_manifest.py --valid-percent=0.0 $DATA_PATH --dest=manifests/train/
```
Next, we need a pre-trained HuBERT-base-ls960 model [[download]](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) and a corresponding kmeans-100 quantizer [[download]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin). Having those we can quantize the dataset:
```
python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path km.bin \
    --acoustic_model_path hubert_base_ls960.pt \
    --layer 6 \
    --manifest_path manifests/train/train.tsv \
    --out_quantized_file_path manifests/train/units
```

Finally, by running
```
python examples/textless_nlp/pgslm/scripts/join_units_manifest.py --manifest=manifests/train/train.tsv --units=manifests/train/units --output=train.txt
```
We will get the training data description `train.txt` in the format that pGSLM expects. The above steps have to be repeated for 
dev/test sets. Importantly, we rely on an assumption that the directories are structured as in LibriSpeech, i.e. the file paths follow the
`<spk_id>/<session_id>/<sample_id>.wav` format.

### Preprocess data for pGSLM
The very first step is to obtain the F0 quantization bins.
Assume the vocoder training manifest is `vocoder_train.txt` (in pGSLM data format prepared with the same process above).
We prepare the quantized F0 from the vocoder training data by running
```sh
bash examples/textless_nlp/pgslm/scripts/prepare_f0_quantization.sh \
  vocoder_train.txt <sample_rate> 32 <preprocessed_dir> <output_prefix> # we use 32 bins in the paper
```
- `<sample_rate>`: sampling rate of the audio files in the manifest
- `<preprocessed_dir>`: where to output the output files
- `<output_prefix>`: prefix of the output files

The script will generate 
- `<output_prefix>.f0_stat.pt`: the speaker-level F0 statistics, which can be used in vocoder training
- `<output_prefix>_mean_norm_log_f0_bin.th`: the quantized F0, which should be used in `prepare_data.sh` below

**Note:** See "Pre-trained models" for the pre-computed speaker-level F0 statistics and quantized F0 bins. We suggest using the pre-computed statistics for the data preparation below in order to take advantage of the pre-trained vocoder for waveform generation.

Next prepare the pGSLM data.
Assume train/valid/test manifests are `{train,valid,test}.txt`.
Here is an example of how to preprocess data:

```sh
bash examples/textless_nlp/pgslm/scripts/prepare_data.sh \
  train.txt valid.txt test.txt <n_unit> <hop_size> <sample_rate> \
  <preprocessed_dir>/<output_prefix>_mean_norm_log_f0_bin.th <preprocessed_dir>
```
- `<n_unit>`: discrete unit vocabulary size (we used a kmeans quantizer with the number of units equal to 100 in the example above)
- `<hop_size>`: downsampling rate relative to the waveform (e.g., 320 for HuBERT units)
- `<sample_rate>`: sampling rate of the audio files in the manifest
- `<preprocessed_dir>`: where to output the preprocessed files

This will create the dataset json config used for the next section at
`<preprocessed_dir>/data_config.json`.

Note that the example script uses only one thread to compute F0, which can take
_very long_ for preprocessing large datasets. It is suggested to distribute
jobs over multiple nodes/processes with `--nshards=x` and `--rank=z` (where z is
in [1, x]) in `preprocess_f0.py`, and set `--nshards_list=x` in
`prepare_data.py` correspondingly to collect sharded F0 data.

Now, everything is ready for training a model.

## Training Multi-Stream Transformer Unit Language Model (MS-TLM)

Below is an example command that trains Multi-Stream Transformer Language Model (MS-TLM) on a prepared dataset:
```bash
DATASET=data_config.json

fairseq-train $DATASET \
  --task=speech_unit_modeling \
  --arch="transformer_ulm_tiny" \
  --criterion=speech_unit_lm_criterion \
  --share-decoder-input-output-embed \
  --dropout=0.1 \
  --attention-dropout=0.1 \
  --optimizer="adam" \
  --adam-betas="(0.9, 0.98)" \
  --clip-norm=1.0 \
  --lr=0.0005 \
  --lr-scheduler="inverse_sqrt" \
  --warmup-updates=4000 \
  --warmup-init-lr=1e-07 \
  --tokens-per-sample=3072 \
  --max-tokens=3072 \
  --update-freq=4 \
  --max-epoch=70 \
  --num-workers=0 \
  --skip-invalid-size-inputs-valid-test \
  --loss-weights="1.0;0.5;0.0" \
  --ignore-f0-input \
  --checkpoint-activations \
  --fp16 \
  --max-target-positions=4096 \
  --stream-shifts="1,1" \
  --log-f0 --normalize-f0-mean --interpolate-f0 \
  --ignore-unused-valid-subsets \
  --discrete-duration --discrete-f0
```

Some of the important parameters that are specific to MS-TLM:
 *  `arch`: specifies the Transformer architecture used. Supported options are:
    * `transformer_ulm_tiny` - a tiny model that can be used for debugging; it has 2 layers, 1 attention head, FFN and embedding dimensions of 64,
    * `transformer_ulm` - a base model with 6 layers, 8 heads, embedding dimension 512, and FFN dimensionality of 2048,
    * `transformer_ulm_big` - the largest model we experiment with in the paper: 12-layer/16 heads, 1024/4096 embedding and FFN dimensions;
 * `loss-weights`: this parameter sets importance weights (must be non-negative) for the components of the loss that correspond to unit, duration, and F0 streams. To turn off a component of the loss, its weight has to be set to 0. For instance, to predict only unit stream the parameter should be set to "1;0;0";
 * `stream-shifts`: specifies relative shifts of the two prosodic streams w.r.t. the unit stream (duration and F0, respectively). No shift corresponds to "0,0";
 * `ignore-duration-input`/`ignore-f0-input`: setting these flags would zero-out correpsonding input streams;
 * `max-token-duration`: duration values would be max-capped by the specified value;
 * `discrete-duration`/`discrete-f0`: whether duration and F0 streams should be quantized;
 * `log_f0`, `normalize-f0-mean`, `normalize-f0-std`, `interpolate-f0`: configure how F0 stream is treated. `log_f0` sets up modelling in the log-space, `normalize-f0-mean`/`normalize-f0-std` control per-speaker normalization, and `interpolate-f0` enables F0 interpolation for unvoiced regions where F0 was set to 0,
 * `mask-dur-prob`, `mask-f0-prob`, `mask-dur-seg-prob`, `mask-f0-seg-prob`, `mask-unit-seg-prob`, `mask-unit-seg-leng`: this family of parameters sets the probababilities of masking individual steps and spans on each stream as well as lengths of the maked spans.


## Pre-trained models
### MS-TLM
Below you can find checkpoints for four best-performing models from the paper (IDs 9..12 in Table 1). These models are trained on Hubert-100 transcripts of the LibriLight-6K dataset. They have the prosody streams shifted by 1 w.r.t. the unit stream. All models predict all three streams (units, duration, and F0), but two
of them only have unit steam in their input.

|                   | Continuous prosody | Quantized prosody |
|-------------------|--------------------|-------------------|
| No prosody input  | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/continuous_no_prosody_shift_1_1.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/discrete_no_prosody_shift_1_1.pt)  |
| Has prosody input | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/continuous_prosody_shift_1_1.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/ulm_checkpoints/discrete_prosody_shift_1_1.pt)|

The optimal per-stream sampling temperatures/scaling parameters that we have identified for these models, in the (`T-token, T-duration, T-f0`) format:

|                   | Continuous prosody | Quantized prosody |
|-------------------|--------------------|-------------------|
| No prosody input  |  0.7, 0.125, 0.0003125|    0.7, 0.25, 0.5 |
| Has prosody input |  0.7, 0.125, 0.00125  |   0.7, 0.25, 0.7  |

## Vocoder
|       Units       | Prosody | F0 stats     | Checkpoint | Config |
|-------------------|---------|--------------|------------|--------|
| HuBERT-base-ls960, kmeans-100 | [[Quantized 32 bins]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_seg_bin.th) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/f0_stats.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/naive_quant_32_norm_log_seg_hubert/checkpoint.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/naive_quant_32_norm_log_seg_hubert/config.json) |
| HuBERT-base-ls960, kmeans-100 | Continuous | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/f0_stats.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_hubert/checkpoint.pt) | [[download]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_hubert/config.json) |


## Evaluating a trained model
Evaluation is done with the `eval/cont_metrics.py` scripts. As described in the paper, there are several metrics used.

**Teacher-forced metrics**
```bash
SET=valid
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
DATA=data_config.json

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
  --metric=teacher_force_everything \
  --path=$CHECKPOINT_PATH \
  --batch-size=16 \
  --fp16 \
  --seed=111 \
  --eval-subset=$SET \
  --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th --dequantize-prosody 
```
(Using this command, our provided `discrete_prosody_shift_1_1.pt` checkpoint should produce `{'token_loss': 1.408..., 'duration_loss': 0.5424..., 'f0_loss': 0.0474...}` on LibriSpeech dev-clean).

The parameters `--f0-discretization-bounds=mean_norm_log_f0_seg_bin.th --dequantize-prosody` are specific for quantized-prosody models. They signal that the prosody streams must be decoded into the continuous domain before calculating correlation. It is the same `*_mean_norm_log_f0_bin.th` file as we prepared before.
The `mean_norm_log_f0_seg_bin.th` file we used with the pre-trained models can be downloaded [[here]](https://dl.fbaipublicfiles.com/textless_nlp/pgslm/vocoder/blizzard2013/mean_norm_log_f0_seg_bin.th).


**Consistency (aka Correlation) metrics**

The following command estimates correlation between mean values of the F0 stream in the prompt and in the generated continuation (unit and duration steams are fixed).

```bash
T_F0=0.7
EXPLOSION=20
SET=test
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
DATA=data_config.json

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
    --prefix-length=150 \
    --metric=correlation \
    --path=$CHECKPOINT_PATH \
    --batch-size=16 \
    --fp16 \
    --seed=111 \
    --teacher-force-tokens \
    --teacher-force-duration  \
    --min-length=300  \
    --batch-explosion-rate=$EXPLOSION \
    --T-f0=$T_F0 \
    --eval-subset=$SET \
    --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th \
    --dequantize-prosody --n-workers=8
```
(Using this command, our provided `discrete_prosody_shift_1_1.pt` checkpoint should produce `{...'F0 corr': 0.315 ..}` on LibriSpeech test-clean).

 * By using flags `--teacher-force-tokens, --teacher-force-duration, --teacher-force-f0` one can calculate correlations along each stream while having other two streams fixed to ground-truth values (or freeze all three streams to get ground-truth correlation values);
 * The parameters `T-f0`, `T-duration`, and `T-token` specify per-stream temperatures and, in the case of continuous-valued prosody, scaling parameter of the corresponding Laplace distribution (setting a temperature to 0 will enforce greedy sampling);
 * `min-length` filters out sequences that are shorter then 300 duration units (i.e. 6s in the case of Hubert units);
 * `prefix-length` specifies that we want to use first 150 duration units are prompt (i.e. 3s in the case of Hubert units)


**Correctness (aka Continuation) and Expressiveness (aka Std) metrics**

By running the following command, we can get minMAE and Std for the log-F0 stream for the model with quantized prosody.
```bash
DATA=data_config.json
EXPLOSION=20
SET=test
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
T_F0=0.7

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
  --prefix-length=150 \
  --metric=continuation \
  --path=$CHECKPOINT_PATH \
  --batch-size=16 \
  --fp16 \
  --seed=111 \
  --batch-explosion-rate=$EXPLOSION \
  --teacher-force-tokens \
  --teacher-force-duration \
  --T-f0=$T_F0 \
  --eval-subset=$SET \
  --f0-discretization-bounds=mean_norm_log_f0_seg_bin.th --dequantize-prosody
```
(Using this command, our provided `discrete_prosody_shift_1_1.pt` checkpoint should produce `{...'F0 MAE': 0.0772, 'F0 Std': 0.1489...}` on LibriSpeech test-clean).

Again, by setting `--teacher-force-tokens, --teacher-force-duration, --teacher-force-f0` we can calculate Token BLEU for the token stream (when `--teacher-force-duration` &  `--teacher-force-f0` are on) and per-stream min MAE for each prosody stream individually.

Finally, `cont_metrics.py` allows to specify the number of workers (e.g., `n-workers=8`) which allows to speed up the computation by spreading multiple worker processes 
over the available GPUs.

**Cont Word BLEU**

We used the code and the evaluation protocol of [(Lakhotia et al., 2021)](https://arxiv.org/abs/2102.01192).

## Sampling from a trained model

To get (prompted or not) samples from a trained model it is enough to run `sample.py`:
```bash
CHECKPOINT_PATH=checkpoints/checkpoint_best.pt
DATASET=examples/textless_nlp/pgslm/repro/dataset/data_config.json 
python examples/textless_nlp/pgslm/sample/sample.py $DATASET \
  --output=$SAMPLES \
  --path=$CHECKPOINT_PATH \
  --sampling \
  --T-token=0.7 \
  --T-duration=0.25 \
  --T-f0=0.7 \
  --max-length=500 \
  --prefix-length=150 \
  --subset=valid \
  --seed=1 \
  --match-duration \
  --code-type=hubert \
  --batch-explosion-rate=2
```

Some useful parameters:
 * `T-token`, `T-duration`, `T-f0` specify sampling temperature for the three streams. Setting a temperature to `0` switches sample to the greedy (argmax) one;
 * `prefix-length`: length of the prompt, measured in timesteps (e.g. for Hubert (CPC) each timestep is 20 (10) ms);
 * `subset`: which subset of the dataset to use as prompts (can be `train`, `valid`, `test`);
 * `teacher-force-tokens`, `teacher-force-duration`, `teacher-force-f0`: if set, at each autoregressive step, ground-truth values replace the produced one;
 * `short-curcuit`: replace sampling by ground-truth inputs;
 * `match-duration`: forces the produced sample to have the same duration (in time), as the entire sequence (beyond the prompt if there is any);
 * `batch-explosion-rate`: number of samples per prompt;
 * `f0-discretization-bounds`: path to a file with quantization boundaries. If it is set, F0 values are de-quantized back to the continuous domain
      (the model must be a quanized one);
  * `max-length` sets the maximal number of segment steps to be produced.

Note that `sample.py` automatically uses all available GPUs, to avoid that please use environment variable `CUDA_VISIBLE_DEVICES`.

## Vocoding samples
To generate audios for output from `sample.py` (`$IN_FILE`):
```bash
python examples/textless_nlp/pgslm/generate_waveform.py \
  --in-file=$IN_FILE \
  --vocoder=$VODOER \
  --vocoder-cfg=$VOCODER_CFG \
  --results-path=$RESULTS_PATH
```
See "Pre-trained model" for `$VOCODER` and `VOCODER_CFG`.
