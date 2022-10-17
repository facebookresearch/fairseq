# ASR-BLEU evaluation toolkit

This toolkit provides a set of public ASR models used for evaluation of different speech-to-speech translation systems at Meta AI. It enables easier score comparisons between different system's outputs.

The ASRGenerator wraps different CTC-based ASR models from HuggingFace and fairseq code bases. Torchaudio CTC decoder is built on top of it to decode given audio files.

Please see `asr_model_cfgs.json` for a list of languages covered currently.

The high-level pipeline is simple by design: given a lang tag, script loads the ASR model, transcribes model's predicted audio, and computes the BLEU score against provided reference translations using sacrebleu.

# Dependencies

Please see `requirements.txt`. 

# Usage examples

This toolkit have been used with:

* Speechmatrix project: https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix.

* Hokkien speech-to-speech translation project: https://github.com/facebookresearch/fairseq/tree/ust/examples/hokkien.

# Standalone run example

High-level example, please substitute arguments per your case:

```bash
python compute_asr_bleu.py --lang <LANG> \
--audio_dirpath <PATH_TO_AUDIO_DIR> \
--reference_path <PATH_TO_REFERENCES_FILE> \
--reference_format txt
```

For more details about arguments please see the script argparser help.