import torch
import fairseq
import json

data2vec = torch.load("/mnt/c/Users/86181/Desktop/checkpoint_423_400000.pt")
cfg = data2vec['cfg']
with open("data2vec.json", 'w') as f:
    f.write(json.dumps(cfg, indent=4, sort_keys=True))