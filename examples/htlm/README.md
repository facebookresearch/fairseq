# HTLM: Hyper-Text Pre-Training and Prompting of Language Models

Paper: [https://arxiv.org/abs/2107.06955](https://arxiv.org/abs/2107.06955)

## Introduction

HTLM is a BART-Large-sized language model trained on a large-scale web crawl using a modified [denoising objective](https://arxiv.org/abs/1910.13461). In our paper, we showed that modeling hyper-text has many advantages: 
- It is easily gathered at scale.
- It provides rich document-level and end-task-adjacent supervision (e.g., class and id attributes often encode document category information)
- It allows for new structured prompting that follows the established semantics of HTML (e.g., to do zero-shot summarization by infilling title tags for a webpage that contains the input text). 
  
We showed that pretraining with a BART-style denoising loss directly on simplified HTML provides highly effective transfer for many end tasks and supervision levels. 

HTLM matches or exceeds the performance of comparably sized text-only LMs for zero-shot prompting and fine-tuning for classification benchmarks while setting new state-of-the-art performance levels for zero-shot summarization. HTLM is also highly effective at auto-prompting itself by simply generating the most likely hyper-text formatting for any available training data.

## Downloading The Model
We download the HTLM model and move it into the correct directory with dictionary
```bash
mkdir htlm && cd htlm
wget -N 'https://dl.fbaipublicfiles.com/htlm_v1/checkpoint.pt'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
cd ..
```

## Prompting

We can load up the HTLM model using the BARTModel interface
```python
from fairseq.models.bart import BARTModel
htlm = BARTModel.from_pretrained('htlm/', checkpoint_file='checkpoint.pt')
htlm.eval()
```

We can then replicate the first figure from our paper

```python
context = """~ south korea on monday announced sweeping tax
reforms , including income and corporate tax cuts
to boost growth by stimulating sluggish private
consumption and business investment .""".replace("\n", " ")

prompt = f"""<html><title> <mask>12 </title><body>{context}</body></html>"""

htlm.fill_mask([prompt], topk=3, beam=5, match_source_len=False, max_len_b=128, lenpen=1)

# Output should start with [[('<html><title> ~ South Korea Announces Tax Reforms To Boost Economic Growth ~ </title><body> ....
```

## Notes
The model is very sensitive to hyper-parameters for sequence generations. So here are a couple of hints. 

- If sampling, topp sampling seems to work best
- By default `max_len` in fairseq `SequenceGenerator` is 256, don't forget to set it to something larger if needed.
- Best beam-sizes for us were between 5-15.

Making this model less sensitive is an active area of research.