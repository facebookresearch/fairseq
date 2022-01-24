# Few-shot Learning with Multilingual Language Models

## Introduction

In this work, we train a family of multilingual generative language models, dubbed XGLM, on a balanced corpus covering a diverse set of languages, and study their few- and zero-shot learning capabilities in a wide range of tasks. Our largest model with 7.5 billion parameters sets new state of the art in few-shot learning on more than 20 representative languages, outperforming GPT-3 of comparable size in multilingual commonsense reasoning (+7.4 accuracy points for 0-shot, +9.4 for 4-shot) and natural language inference (+5.4 for 0-shot, +5.4 for 4-shot). We have included a [model card](model_card.md) of XGLM for transparency and accountability.

## Data and Languages
XGLM models are trained on a new multilingual corpus extracted from CommonCrawl (CC100-XL), a significantly larger multilingual dataset covering 68 Common Crawl (CC) snapshots (from [Summer 2013](http://commoncrawl.org/2013/11/new-crawl-data-available/) to [March/April 2020](https://commoncrawl.org/2020/04/march-april-2020-crawl-archive-now-available/) consisting of 134 languages. The detailed languages and data statistics are reported in the paper (Table A.1).

## Pre-trained models

Model | Layers | Model Dim | Languages | Download
---|---|---|---|---
`XGLM 564M` | 24 | 1024 | trained on 30 languages|  [xglm.564M.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.564M.tar.gz)
`XGLM 1.7B` | 24 | 2048 | trained on 30 languages|  [xglm.1.7B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.1.7B.tar.gz)
`XGLM 2.9B` | 48 | 2048 | trained on 30 languages|  [xglm.2.9B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.2.9B.tar.gz)
`XGLM 7.5B` | 32 | 4096 | trained on 30 languages|  [xglm.7.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.7.5B.tar.gz)
`XGLM 4.5B` | 48 | 2048 | trained on 134 languages|  [xglm.4.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.4.5B.tar.gz)

## Evaluation

### Example (COPA)

The following snippet show how to evaluate our models on the Choice of Plausible Alternatives (COPA) task, using examples in English, Chinese and Hindi. 

```python
data_samples = {
    'en': [
        {
            "premise": "I wanted to conserve energy.", 
            "choice1": "I swept the floor in the unoccupied room.", 
            "choice2": "I shut off the light in the unoccupied room.",
            "question": "effect",
            "label": "1"
        },
        {
            "premise": "The flame on the candle went out.",
            "choice1": "I blew on the wick.", 
            "choice2": "I put a match to the wick.",
            "question": "cause",
            "label": "0"
        }
    ],
    'zh': [
        {
            "premise": "我想节约能源。", 
            "choice1": "我在空着的房间里扫了地板。", 
            "choice2": "我把空房间里的灯关了。",
            "question": "effect",
            "label": "1"
        },
        {
            "premise": "蜡烛上的火焰熄灭了。",
            "choice1": "我吹灭了灯芯。", 
            "choice2": "我把一根火柴放在灯芯上。",
            "question": "cause",
            "label": "0"
        }
    ],
    'hi': [
        {
            "premise": "M te vle konsève enèji.", 
            "choice1": "Mwen te fin baleye chanm lib la.", 
            "choice2": "Mwen te femen limyè nan chanm lib la.",
            "question": "effect",
            "label": "1"
        },
        {
            "premise": "Flam bouji a te etenn.",
            "choice1": "Mwen te soufle bouji a.", 
            "choice2": "Mwen te limen mèch bouji a.",
            "question": "cause",
            "label": "0"
        }
    ]
}
```
In this example, we format the examples use the non-verbal prompts `{premise}\n{choice1}` and `{premise}\n{choice2}`, which are shared by all three languages. 
```python
from fairseq.models.transformer_lm import TransformerLanguageModel

model_dir = 'path_to_decompressed_tar_gz_dir'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    
# Zero-shot evaluation for the Choice of Plausible Alternatives (COPA) task.
# A return value of 0 indicates that the first alternative is more plausible,
# while 1 indicates that the second alternative is more plausible.
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 0 if lprob1 > lprob2 else 1
    
for lang in ['en', 'zh', 'hi']:
    for idx, example in enumerate(data_samples[lang]):
        predict = COPA_eval(example["premise"], example["choice1"], example["choice2"])
        print(f'{lang}-{idx}', predict, example['label'])
        
# en-0 1 1
# en-1 0 0
# zh-0 1 1
# zh-1 0 0
# hi-0 1 1
# hi-1 0 0
```

## Citation
```
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
