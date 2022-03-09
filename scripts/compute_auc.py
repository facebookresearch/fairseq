from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.data import Dictionary
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

import os
import torch

# bbbp model = "/fsx-html2/armenag/chemical/checkpoints/ft.bart_large.sentpred.ms16.uf1.mu2296.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm367.fp16.ngpu8/"
# dataset = "bbbp"

# model = "/fsx-html2/armenag/chemical/checkpoints/ft_clintox.bart_large.sentpred.ms16.uf1.mu2296.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm367.fp16.ngpu8"
# dataset = "clintox"

model = "/fsx-html2/armenag/chemical/checkpoints/ft_tox21.bart_large.sentpred.ms16.uf1.mu2296.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm367.fp16.ngpu8/"
dataset = "tox21"

os.system(f"mkdir {model}/input0")
os.system(f"mkdir {model}/label")

os.system(f"cp /data/home/armenag/code/chem/evaluation_data/{dataset}/processed/input0/dict.txt {model}/input0/")
os.system(f"cp /data/home/armenag/code/chem/evaluation_data/{dataset}/processed/label/dict.txt {model}/label/")

bart = BARTModel.from_pretrained(model, checkpoint_file='checkpoint_last.pt', bpe="sentencepiece",
                                 sentencepiece_model="/fsx-html2/armenag/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda()

input_dict = Dictionary.load(f"/data/home/armenag/code/chem/evaluation_data/{dataset}/processed/input0/dict.txt")
input_dict.add_symbol("<mask>")
target_dict = Dictionary.load(f"/data/home/armenag/code/chem/evaluation_data/{dataset}/processed/label/dict.txt")

smiles = list(load_indexed_dataset(
    f"/data/home/armenag/code/chem/evaluation_data/{dataset}/processed/input0/test", input_dict))
targets = list(load_indexed_dataset(
    f"/data/home/armenag/code/chem/evaluation_data/{dataset}/processed/label/test", target_dict))

y_pred = []
y = []
for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets)))):
    smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
    target = target[0].item()
    y_pred.append(bart.predict('sentence_classification_head', smile)[0][1].exp().item())
    y.append(target - 4)

print("ROC_AUC_SCORE: ", roc_auc_score(y, y_pred))
print(classification_report(y, [int(x+0.5) for x in y_pred]))
