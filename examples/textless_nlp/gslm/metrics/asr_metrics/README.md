# ASR-based evaluation

Overall, the life cycle of the ASR-based evaluation for an ULM contains the following steps:
 1. Training an ULM and sampling from it [[description]](./../../ulm)
 2. Running UTS on the sampled unit sequences [[description]](./../../unit2speech)
 3. Pre-processing for the ASR (down-sampling to 16 KHz, aligning length of the generated audio with ground-truth utterances)
 4. Running ASR
 5. Calculation of the post-ASR evaluation metrics

Here we assume that you have already went throught the first two steps and focus on the rest.

## Preprocessing
### Down-sampling to 16KHz
The bulk conversion can be done by running
```bash
 python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/unit2speech/convert_to_16k.py $UTS_OUTPUT $UTS_OUTPUT_DOWNSAMPLE
 ```
 where `$UTS_OUTPUT` specifies the directory with the generated audio and `$UTS_OUTPUT_DOWNSAMPLE` is the directory where downsampled audio would be saved.

 ### Matching by length
This step is somewhat optional. However, if you want to compare the fluency and diversity of a generated speech utterance to that of the ground-truth speech with the same prefix, it is a good idea to force them to be of the same length.
```bash
python $FAIRSEQ_ROOT/examples/textless_nlp/asr_metrics/cut_as.py \
    --samples_dir=$UTS_OUTPUT_DOWNSAMPLE --out_dir=$UTS_OUTPUT_DOWNSAMPLE_CUT \
    --prompts_description=data/ground_truth_continuation_dev.json
```

Here `ground_truth_continuation_dev.json` is a json file with ground-truth text from LibriSpeech dev-clean, associated with some meta-data (assuming the evaluation is done on dev-clean). This file can be downloaded [[here]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/ground_truth_continuation_dev.json). A similar file for the test-clean is [[here]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/ground_truth_continuation_test.json). These files are used for the evaluation and contain texts for audio sequences that are at least 6s long.

## Running ASR
We use a pre-trained wav2vec model to run the ASR step. We firstly need to prepare manifest files which, roughly, tell the ASR system which files we want to transcribe. You can find more details and download the `960h_scratch.pt` checkpoint
[[here]](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md)). To run ASR, you would also need to
install KenLM, Flashlight decoder, and download the KenLM 4-gram English language model.

```bash
 python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py  \
    $UTS_OUTPUT_DOWNSAMPLE_CUT --valid-percent 0.0  --dest $MANIFEST_DIR --ext wav
```
where `$UTS_OUTPUT_DOWNSAMPLE_CUT` speficies the directory with the preprocessed UTS outputs and `$MANIFEST_DIR` is the output directory.

We will be running an out-of-the-box evaluation script which requires ground-truth transcripts to measure quality metrics. We are only
interested in the transcripts (and we don't have ground-truth outputs for when our ULM generated!), hence we will just generate
some dummy transcripts instead:
```bash
cp $FAIRSEQ_ROOT/examples/textless_nlp/gslm/asr_metrics/misc/dict.ltr.txt $MANIFEST_DIR
python $FAIRSEQ_ROOT/examples/textless_nlp/gslm/asr_metrics/misc/dummy_asr_data.py  --tsv=$MANIFEST_DIR/train.tsv \
 --output-dir=$MANIFEST_DIR
```

Now we are ready for running ASR:
```
mkdir -p asr
python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py  \
    $MANIFEST_DIR \
    --task audio_pretraining --nbest 1 --path 960h_scratch.pt \
    --gen-subset=train --results-path $PATH_TO_ASR_OUTPUT \
    --w2l-decoder kenlm --lm-model 4-gram.bin \
    --lexicon librispeech/lexicon_ltr.lst --word-score -1 \
    --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 300000 --remove-bpe letter
```
where `lexicon_ltr.lst` is the LibriSpeech lexicon and `$PATH_TO_ASR_OUTPUT` is the output directory (can be downloaded [[here]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst)).

## Evaluation metrics
We run evaluation on the 1_000 shortest sequences that are at least 6s long. To filter those from the ASR transcript, we additionally provide each metric script with the paths to the manifest and `ground_truth_continuation_*` files.

### Perplexity (PPX)
To get a PPX metric estimate on an ASR transcript, you need to run the following command:
```bash
python ppx.py $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt --cut-tail\
  --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```
where `--cut-tail` tells the script to ignore the last token on each line (ASR puts the sequence ID there).

### Self- and Auto-BLEU
```bash
python self_bleu.py $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt  --cut-tail \
  --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```

### Continuation-BLEU
```bash
python continuation_eval.py --asr-transcript $PATH_TO_ASR_OUTPUT/hypo.word-960h_scratch.pt-train.txt \
   --manifest=$MANIFEST_DIR/train.tsv --prompts-description=data/ground_truth_continuation_dev.json
```

### AUC
Based on the metrics calculated above, we can estimate the AUC of the perplexity/diversity trade-off. We provide an illustration in a [Colab notebook](https://colab.research.google.com/drive/1pVPfOVax_PU3MkYdHRSsa-SI8GBUldNt?usp=sharing).
