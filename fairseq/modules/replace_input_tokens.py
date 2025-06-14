"""
Simple perturbations described in Rethinking Perturbations in Encoder-Decoders for Fast Training
Paper: https://aclanthology.org/2021.naacl-main.460/
Original repo: https://github.com/takase/rethink_perturbations
"""

import torch
import torch.nn as nn

#functions for input token replacement
def sample_tokens(base_emb, embed_module):
    #sampling with gumbel max trick
    logit = nn.functional.log_softmax(nn.functional.linear(base_emb, embed_module.weight), dim=-1)
    noise = logit.clone()
    maxind = (logit - (-noise.uniform_().log()).log()).argmax(dim=-1)
    return maxind.detach()

def replace_input_tokens(base_emb, embed_module, replace_rate, sampling_method, decoder=False):
    #shape of base emb: B x T x C
    #embed_module: torch module embedding
    #select keep tokens
    keep_flag = base_emb.new_full((base_emb.size(0), base_emb.size(1)), 1 - replace_rate)
    keep_flag = keep_flag.bernoulli()
    if decoder:
        #keep bos tag in decoder side
        keep_flag[:, 0] = 1
    if sampling_method == 'worddrop':
        return base_emb * keep_flag.unsqueeze(2)
    elif sampling_method == 'similarity':
        #replace embeddings with other token embedding using gumbel max trick
        keep_flag = keep_flag.unsqueeze(2)
        #sampling embeddings for replacement
        sampled_tokens = sample_tokens(base_emb, embed_module)
        sampled_embeds = embed_module(sampled_tokens)
        return base_emb * keep_flag + sampled_embeds * (1 - keep_flag)
    elif sampling_method == 'uniform':
        #replace embeddings based on uniform distribution
        keep_flag = keep_flag.unsqueeze(2)
        #uniform sampling for ward index
        uniform = base_emb.new_empty((base_emb.size(0), base_emb.size(1), embed_module.weight.size(-1)))
        uniform.uniform_()
        maxind = uniform.argmax(dim=-1).detach()
        sampled_embeds = embed_module(maxind)
        return base_emb * keep_flag + sampled_embeds * (1 - keep_flag)
