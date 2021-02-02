# Flashlight Decoder

This script runs decoding for pre-trained speech recognition models.

## Usage

Assuming a few variables:

```bash
exp_dir=<path-to-experiment-directory>
data=<path-to-data-directory>
lm_model=<path-to-language-model>
lexicon=<path-to-lexicon>
```

Example usage for decoding a fine-tuned Wav2Vec model:

```bash
python $FAIRSEQ_ROOT/examples/speech_recognition/hydra/infer.py --multirun \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    decoding.exp_dir=$exp_dir \
    decoding.decoder.name=kenlm \
    decoding.decoder.lexicon=$lexicon \
    decoding.decoder.lmpath=$lm_model \
    dataset.gen_subset=dev_clean,dev_other,test_clean,test_other
```

Example usage for using Ax to sweep WER parameters (requires `pip install hydra-ax-sweeper`):

```bash
python $FAIRSEQ_ROOT/examples/speech_recognition/hydra/infer.py --multirun \
    hydra/sweeper=ax \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    decoding.exp_dir=$exp_dir \
    decoding.decoder.name=kenlm \
    decoding.decoder.lexicon=$lexicon \
    decoding.decoder.lmpath=$lm_model \
    decoding.write_sentences=false \
    decoding.unique_wer_file=true \
    dataset.gen_subset=dev_other
```
