# MMS: Scaling Speech Technology to 1000+ languages

The Massively Multilingual Speech (MMS) project expands speech technology from about 100 languages to over 1,000 by building a single multilingual speech recognition model supporting over 1,100 languages (more than 10 times as many as before), language identification models able to identify over [4,000 languages](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html) (40 times more than before), pretrained models supporting over 1,400 languages, and text-to-speech models for over 1,100 languages. Our goal is to make it easier for people to access information and to use devices in their preferred language.  

You can find details in the paper [Scaling Speech Technology to 1000+ languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/) and the [blog post](https://ai.facebook.com/blog/multilingual-model-speech-recognition/).

An overview of the languages covered by MMS can be found [here](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html).


## Pretrained models

| Model | Link
|---|---
MMS-300M | [download](https://dl.fbaipublicfiles.com/mms/pretraining/base_300m.pt)
MMS-1B | [download](https://dl.fbaipublicfiles.com/mms/pretraining/base_1b.pt)

Example commands to finetune the pretrained models can be found [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#fine-tune-a-pre-trained-model-with-ctc).

## Finetuned models
### ASR

| Model | Languages | Dataset | Model | Supported languages |
|---|---|---|---|---
MMS-1B:FL102 | 102 | FLEURS | [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_fl102.pt) | [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_fl102_langs.html) 
MMS-1B:L1107| 1107 | MMS-lab | [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107.pt) | [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_l1107_langs.html) 
MMS-1B-all| 1162 | MMS-lab + FLEURS <br>+ CV + VP + MLS |  [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all.pt) | [download](https://dl.fbaipublicfiles.com/mms/asr/mms1b_all_langs.html)

### TTS
1. Download the list of [iso codes](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) of 1107 languages.
2. Find the iso code of the target language and download the checkpoint. Each folder contains 3 files: `G_100000.pth`,  `config.json`, `vocab.txt`. The `G_100000.pth` is the generator trained for 100K updates, `config.json` is the training config, `vocab.txt` is the vocabulary for the TTS model. 
```
# Examples:
wget https://dl.fbaipublicfiles.com/mms/tts/eng.tar.gz # English (eng)
wget https://dl.fbaipublicfiles.com/mms/tts/azj-script_latin.tar.gz # North Azerbaijani (azj-script_latin)
```

### LID

\# Languages | Dataset | Model | Dictionary | Supported languages |
|---|---|---|---|---
126 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l126.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l126/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l126_langs.html)
256 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l256.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l256/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l256_langs.html)
512 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l512.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l512/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l512_langs.html)
1024 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l1024.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l1024/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l1024_langs.html)
2048 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l2048.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l2048/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l2048_langs.html)
4017 | FLEURS + VL + MMS-lab-U + MMS-unlab | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l4017.pt) | [download](https://dl.fbaipublicfiles.com/mms/lid/dict/l4017/dict.lang.txt) | [download](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l4017_langs.html)

## Commands to run inference 

### ASR
Run this command to transcribe one or more audio files:
```shell command
cd /path/to/fairseq-py/
python examples/mms/asr/infer/mms_infer.py --model "/path/to/asr/model" --lang lang_code --audio "/path/to/audio_1.wav" "/path/to/audio_1.wav"
```

For more advance configuration and calculate CER/WER, you could prepare manifest folder by creating a folder with this format: 
```
$ ls /path/to/manifest
dev.tsv
dev.wrd
dev.ltr
dev.uid

# dev.tsv each line contains <audio>  <number_of_sample>
$ cat dev.tsv
/
/path/to/audio_1  180000
/path/to/audio_2  200000

$ cat dev.ltr
t h i s | i s | o n e |
t h i s | i s | t w o |

$ cat dev.wrd
this is one
this is two

$ cat dev.uid
audio_1
audio_2
```

Followed by command below:
```
lang_code=<iso_code>

PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m --config-dir examples/mms/config/ --config-name infer_common decoding.type=viterbi dataset.max_tokens=4000000 distributed_training.distributed_world_size=1 "common_eval.path='/path/to/asr/model'" task.data='/path/to/manifest' dataset.gen_subset="${lang_code}:dev" common_eval.post_process=letter

```
Available options:
* To get the raw character-based output, user can change to `common_eval.post_process=none` 

* To maximize GPU efficiency or avoid out-of-memory (OOM), user can tune `dataset.max_tokens=???` size

* To run language model decoding, install flashlight python bindings using
  ```
  git clone --recursive git@github.com:flashlight/flashlight.git
  cd flashlight; 
  git checkout 035ead6efefb82b47c8c2e643603e87d38850076 
  cd bindings/python 
  python3 setup.py install
  ```
  Train a [KenLM language model](https://github.com/flashlight/wav2letter/tree/main/recipes/rasr#language-model) and prepare a lexicon file in [this](https://dl.fbaipublicfiles.com/wav2letter/rasr/tutorial/lexicon.txt) format. 
  ```
   LANG=<iso> # for example - 'eng', 'azj-script_latin'
   PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1  python examples/speech_recognition/new/infer.py  --config-dir=examples/mms/asr/config \
      --config-name=infer_common decoding.type=kenlm  distributed_training.distributed_world_size=1  \ 
      decoding.unique_wer_file=true   decoding.beam=500 decoding.beamsizetoken=50  \
      task.data=<MANIFEST_FOLDER_PATH>   common_eval.path='<MODEL_PATH.pt>' decoding.lexicon=<LEXICON_FILE> decoding.lmpath=<LM_FILE> \  
      decoding.results_path=<OUTPUT_DIR> dataset.gen_subset=${LANG}:dev decoding.lmweight=??? decoding.wordscore=???
  ```
   We typically sweep `lmweight` in the range of 0 to 5 and `wordscore` in the range of -3 to 3.  The output directory will contain the reference and hypothesis outputs from decoder. 
   
   For decoding with character-based language models, use empty lexicon file (`decoding.lexicon=`), `decoding.unitlm=True` and sweep over `decoding.silweight` instead of `wordscore`. 

### TTS
Note: clone and install [VITS](https://github.com/jaywalnut310/vits) before running inference.
```shell script
## English TTS
$ PYTHONPATH=$PYTHONPATH:/path/to/vits python examples/mms/tts/infer.py --model-dir /path/to/model/eng \
--wav ./example.wav --txt "Expanding the language coverage of speech technology \
has the potential to improve access to information for many more people"

## Maithili TTS
$ PYTHONPATH=$PYTHONPATH:/path/to/vits python examples/mms/tts/infer.py --model-dir /path/to/model/mai \
--wav ./example.wav --txt "मुदा आइ धरि ई तकनीक सौ सं किछु बेसी भाषा तक सीमित छल जे सात हजार \ 
सं बेसी ज्ञात भाषाक एकटा अंश अछी"
```
`example.wav` contains synthesized audio for the language.


### LID


Prepare two files in this format 
```
#/path/to/manifest.tsv
/
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav

# /path/to/manifest.lang
eng 1
eng 1
eng 1
```

Download model and the corresponding dictionary file for the LID model. 
Use the following command to run inference - 
```shell script
$  PYTHONPATH='.'  python3  examples/mms/lid/infer.py /path/to/dict/l126/ --path /path/to/models/mms1b_l126.pt \
  --task audio_classification  --infer-manifest /path/to/manifest.tsv --output-path <OUTDIR>
```
The above command assumes there is a file named `dict.lang.txt` in `/path/to/dict/l126/`. `<OUTDIR>/predictions.txt` will contain the predictions from the model for the audio files in `manifest.tsv`. 


## Forced Alignment Tooling

We also developed an efficient forced alignment algorithm implemented on GPU which is able to process very long audio files. This algorithm is open sourced and we provide instructions on how to use it [here](data_prep). We also open source a multilingual alignment model trained on 31K hours of data in 1,130 languages, as well as text normalization scripts.


# License

The MMS code and model weights are released under the CC-BY-NC 4.0 license.

# Citation

**BibTeX:**

```
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
  journal={arXiv},
  year={2023}
}

```
