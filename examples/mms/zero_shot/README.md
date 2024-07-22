# MMS Zero-shot Speech Recognition

This project builds a single multilingual speech recognition for almost **all** languages spoken in the world by leveraging uroman text as intermediate representation. The model is pre-trained on supervised data in over 1000 languages. At inference time, one only needs to build lexicon and and optional N-gram language models for the unseen language.

You can download the zero-shot uroman model [here](https://dl.fbaipublicfiles.com/mms/zeroshot/model.pt) and dictionary [here](https://dl.fbaipublicfiles.com/mms/zeroshot/tokens.txt)

Checkout the demo here [![Open In HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/mms-meta/mms-zeroshot) 

## Commands to run inference

1. Prepare uroman-based lexicon: build a lexicon file by applying [uroman](https://github.com/isi-nlp/uroman) over your file or words. Refer to the format below. 
```
abiikira a b i i k i r a |
úwangaba u w a n g a b a |
banakana b a n a k a n a |
...
...
```

Each uroman token in the spelling of the final lexicon should appear in the token dictionary of the model.

2. [Optional] Prepare N-gram language model: build LMs with [KenLM](https://github.com/kpu/kenlm). We found using even 1-gram LMs can produce very good results.

Inference command example

```
# lexicon only

model_path= # place the downloaded uroman model here
data_path= # path containing your tsv and wrd files
subset= # subset in your data path
lex_filepath= # your uroman lexicon
lm_filepath= # any n-gram lm as a placeholder; not used
wrdscore=-3.5 # can be tuned on your data
res_path=
bs=2000 # bs=500 is good too

PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m             --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=kenlm             dataset.max_tokens=2000000 distributed_training.distributed_world_size=1             "common_eval.path=${model_path}" task.data=${data_path}             dataset.gen_subset=mms_eng:${subset} decoding.lexicon=${lex_filepath}             decoding.lmpath=${lm_filepath} decoding.lmweight=0 decoding.wordscore=${wrdscore}    decoding.silweight=0 decoding.results_path=${res_path}             decoding.beam=${beam}
```


```
# n-gram lm

model_path= # place the downloaded uroman model here
data_path= # path containing your tsv and wrd files
subset= # subset in your data path
lex_filepath= # your uroman lexicon
lm_filepath= # your kenlm
wrdscore=-0.18 # wrdscore and lmweight can be tuned together on your data
lmweight=1.48
res_path=
bs=2000

PYTHONPATH=. PREFIX=INFER HYDRA_FULL_ERROR=1 python examples/speech_recognition/new/infer.py -m             --config-dir examples/mms/asr/config/ --config-name infer_common decoding.type=kenlm             dataset.max_tokens=2000000 distributed_training.distributed_world_size=1             "common_eval.path=${model_path}" task.data=${data_path}             dataset.gen_subset=mms_eng:${subset} decoding.lexicon=${lex_filepath}             decoding.lmpath=${lm_filepath} decoding.lmweight=${lmweight} decoding.wordscore=${wrdscore}             decoding.silweight=0 decoding.results_path=${res_path}             decoding.beam=${bs}

```

Note that the commands won't give proper CER directly, as they don't handle your reference file properly if your script is not included the dictionary. You will need to calculate CER yourself after generation is done.

# License

The MMS Zero shot code and model weights are released under the CC-BY-NC 4.0 license.

# Citation

**BibTeX:**

```
@article{zhao2024zeroshot,
  title={Scaling a Simple Approach to Zero-shot Speech Recognition},
  author={Jinming Zhao, Vineel Pratap and Michael Auli},
  journal={arXiv},
  year={2024}
}

```
