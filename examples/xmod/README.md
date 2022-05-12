# X-MOD: Lifting the Curse of Multilinguality by Pre-training Modular Transformers

X-MOD extends multilingual masked language models like XLM-R to include language-specific modular components, introduced at each transformer layer. Each module is only used by one language. For fine-tuning, the modular components are frozen, and replaced with the target language in cross-lingual transfer settings.


## Pre-trained models

Model | Size | # train steps | # langs | Download
---|---|---|---|---
`xmod.base.13.125k` | BERT-base | 125k | 13 | [xmod.base.13.125k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.13.125k.tar.gz)
`xmod.base.30.125k` | BERT-base | 125k | 30 | [xmod.base.30.125k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.30.125k.tar.gz)
`xmod.base.30.195k` | BERT-base | 195k | 30 | [xmod.base.30.195k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.30.195k.tar.gz)
`xmod.base.60.125k` | BERT-base | 125k | 60 | [xmod.base.60.125k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.60.125k.tar.gz)
`xmod.base.60.265k` | BERT-base | 265k | 60 | [xmod.base.60.265k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.60.265k.tar.gz)
`xmod.base.75.125k` | BERT-base | 125k | 75 | [xmod.base.75.125k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.75.125k.tar.gz)
`xmod.base.75.269k` | BERT-base | 269k | 75 | [xmod.base.75.269k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.75.269k.tar.gz)
`xmod.base` | BERT-base | 1M | 81 | [xmod.base.81.1M.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.base.81.1M.tar.gz)
`xmod.large.prenorm` | BERT-large | 500k | 81 | [xmod.large.prenorm.81.500k.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xmod/xmod.large.prenorm.81.500k.tar.gz)


## Fine-tuning on NLI

We next provide an example of how to fine-tune the pre-trained models above on Natural Language Inference (NLI). We use MNLI for training in English, and show how to run inference in other languages.

### 1) Download a pre-trained model

```bash
MODEL=xmod.base.81.1M
wget https://dl.fbaipublicfiles.com/fairseq/models/xmod/$MODEL.tar.gz
tar -xzf $MODEL.tar.gz
```

### 2) Download and preprocess [MNLI](https://cims.nyu.edu/~sbowman/multinli/)
```bash
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
python ./examples/xmod/preprocess_nli.py \
    --sentencepiece-model $MODEL/sentencepiece.bpe.model \
    --train multinli_1.0/multinli_1.0_train.jsonl \
    --valid multinli_1.0/multinli_1.0_dev_matched.jsonl \
    --destdir multinli_1.0/fairseq
```

### 3) Fine-tune on MNLI:

```bash
MAX_EPOCH=5
LR=1e-05
BATCH_SIZE=32
DATA_DIR=multinli_1.0/fairseq/bin

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $MODEL/model.pt  \
    --save-dir $MODEL/nli \
    --reset-optimizer  \
    --reset-dataloader  \
    --reset-meters  \
    --best-checkpoint-metric accuracy  \
    --maximize-best-checkpoint-metric  \
    --task sentence_prediction_adapters  \
    --num-classes 3  \
    --init-token 0  \
    --separator-token 2   \
    --max-positions 512  \
    --shorten-method "truncate"  \
    --arch xmod_base  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --weight-decay 0.01  \
    --criterion sentence_prediction_adapters  \
    --optimizer adam  \
    --adam-betas '(0.9, 0.98)'  \
    --adam-eps 1e-06  \
    --clip-norm 0.0  \
    --lr-scheduler fixed  \
    --lr $LR \
    --fp16  \
    --fp16-init-scale 4  \
    --threshold-loss-scale 1  \
    --fp16-scale-window 128  \
    --batch-size $BATCH_SIZE  \
    --required-batch-size-multiple 1  \
    --update-freq 1  \
    --max-epoch $MAX_EPOCH
```

### 4) Run inference

After training the model, we can load it and run inference in our target language. The default language is set to English, which is why we were not required to pass a language ID to the model during fine-tuning. To run inference in a non-English language, we need to tell the model that the module of the target language should be used instead:

```python
from fairseq.models.xmod import XMODModel

MODEL='xmod.base.81.1M/nli'
DATA='multinli_1.0/fairseq/bin'

# Load model
model = XMODModel.from_pretrained(
            model_name_or_path=MODEL,
            checkpoint_file='checkpoint_best.pt', 
            data_name_or_path=DATA, 
            suffix='', 
            criterion='cross_entropy', 
            bpe='sentencepiece',  
            sentencepiece_model=DATA+'/input0/sentencepiece.bpe.model')
model = model.eval();  # disable dropout
model = model.half();  # use FP16
model = model.cuda();  # move to GPU

def predict(premise, hypothesis, lang):
    tokens = model.encode(premise, hypothesis)
    idx = model.predict('sentence_classification_head', tokens, lang_id=[lang]).argmax().item()
    dictionary = model.task.label_dictionary
    return dictionary[idx + dictionary.nspecial]

predict(
    premise='X-Mod hat spezifische Module die für jede Sprache existieren.',
    hypothesis='X-Mod hat Module.',
    lang='de_DE'
)  # entailment

predict(
    premise='Londres es la capital del Reino Unido.',
    hypothesis='Londres está en Francia.',
    lang='es_XX',
)  # contradiction

predict(
    premise='Patxik gogoko ditu babarrunak.',
    hypothesis='Patxik babarrunak bazkaldu zituen.',
    lang='eu_ES',
)  # neutral
```
