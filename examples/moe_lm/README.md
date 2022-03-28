# Efficient Large Scale Language Modeling with Mixtures of Experts

## Introduction

Mixture of Experts layers (MoEs) enable efficient scaling of language models
through conditional computation. This work empirically compares how
autoregressive MoE language models scale in comparison with dense models in a
wide range of settings: in- and out-of-domain language modeling, zero- and
few-shot priming, and full fine-tuning. See the associated paper for more
details.

This repo contains instructions for reproducing results from the paper.

## Pre-trained models

These models are intended for research purposes only in order to reproduce the
results from the paper, and to enable further research on the capabilities and
limitations of language models. Please see the [model card](model_card.md) for
more details about how the models were trained and evaluated, as well as their
limitations and intended use.

#### Dense models

Dense models can be run directly from the `main` branch.

Model | Layers | Model Dim | Languages | Download
---|---|---|---|---
`dense_125m` | 12 | 768 | English | [en_dense_lm_125m.tar.gz (0.2GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_125m.tar.gz)
`dense_355m` | 24 | 1024 | English | [en_dense_lm_355m.tar.gz (0.6GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_355m.tar.gz)
`dense_1_3b` | 24 | 2048 | English | [en_dense_lm_1_3b.tar.gz (2.3GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_1_3b.tar.gz)
`dense_2_7b` | 32 | 2560 | English | [en_dense_lm_2_7b.tar.gz (4.6GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_2_7b.tar.gz)
`dense_6_7b` | 32 | 4096 | English | [en_dense_lm_6_7b.tar.gz (12GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_6_7b.tar.gz)
`dense_13b` | 40 | 5120 | English | [en_dense_lm_13b.tar.gz (23GB)](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_dense_lm_13b.tar.gz)

#### Mixture of expert models

MoE models must be run from the `moe` branch. Please see the
[MoE README](https://github.com/pytorch/fairseq/tree/moe#evaluating-moe-language-models)
for more details about how to load and evaluate MoE models.

Model | Layers | Model Dim | Languages | Download
---|---|---|---|---
`moe_15b` | 12 | 768 | English | [en_moe_lm_15b.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_moe_lm_15b.tar.gz)
`moe_52b` | 24 | 1024 | English | [en_moe_lm_52b.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/lm/en_moe_lm_52b.tar.gz)
`moe_207b` | 24 | 2048 | English | Available by request
`moe_1_1t` | 32 | 4096 | English | Available by request

## Evaluation

### Example (COPA)

The following snippet shows how to evaluate our dense models on the [Choice of
Plausible Alternatives (COPA)](https://people.ict.usc.edu/~gordon/copa.html) task.

```python
from fairseq.models.transformer_lm import TransformerLanguageModel
model_dir = '/path/to/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')
lm = lm.eval();  # disable dropout
lm = lm.half();  # use FP16 for evaluation
lm = lm.cuda();  # move to GPU

def get_logprobs(prompt):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']

# Zero-shot evaluation for the Choice of Plausible Alternatives (COPA) task.
# A return value of 1 indicates that the first alternative is more plausible,
# while 2 indicates that the second alternative is more plausible.
def COPA_eval(prompt, alternative1, alternative2):
    lprob1 = get_logprobs(prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(prompt + "\n" + alternative2).sum()
    return 1 if lprob1 > lprob2 else 2

COPA_eval("The man broke his toe. What was the CAUSE of this?", "He got a hole in his sock.", "He dropped a hammer on his foot.")
# 2
COPA_eval("I tipped the bottle. What happened as a RESULT?", "The liquid in the bottle froze.", "The liquid in the bottle poured out.")
# 2
COPA_eval("I knocked on my neighbor's door. What happened as a RESULT?", "My neighbor invited me in.", "My neighbor left his house.")
# 1
```

### Data format

Few-shot prompting is known to be sensitive to the input formatting, and it is usually best to match the formatting used in pretraining.

During pretraining our models were presented with data in the following format (i.e., one paragraph per line, with a blank line separating documents):
```
<doc0,para0,tok0> ... <doc0,para0,tokX>
<doc0,para1,tok0> ... <doc0,para1,tokY>

<doc1,para0,tok0> ... <doc0,para0,tokX>
...
```

#### Newlines

While we use the byte-level BPE from GPT-2/3, fairseq's preprocessing replaces newlines with the end-of-sentence symbol (`</s>`), which corresponds to embedding index `2`.
Thus **the model never saw newline characters during pretraining** and newlines should not be used during few-shot prompting.

This is more clearly illustrated in the following example, which uses fairseq's Hub Interface to tokenize two documents in the desired format:
```python
from fairseq.models.transformer_lm import TransformerLanguageModel
model_dir = '/path/to/en_dense_lm_125m'
lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='gpt2')

data = """\
This is the first paragraph of the first document.
This is the second paragraph of the first document.

This is the first paragraph of the second document.\
"""

# The following is wrong, since it will encode newlines present in `data`.
tokens_bad = lm.score(data)['tokens']
assert '\n' in lm.decode(tokens_bad)  # oops, we encoded a newline

# Instead pass the replace_newlines_with_eos option to get the correct behavior.
tokens_good = lm.score(data, replace_newline_with_eos=True)['tokens']
assert '\n' not in lm.decode(tokens_good)  # no newlines were encoded
```

## Citation

Coming soon.
