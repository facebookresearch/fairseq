# Simple and Effective Noisy Channel Modeling for Neural Machine Translation (Yee et al., 2019)
This page contains pointers to pre-trained models as well as instructions on how to run the reranking scripts.

## Citation:
```bibtex
@inproceedings{yee2018simple,
  title = {Simple and Effective Noisy Channel Modeling for Neural Machine Translation},
  author = {Kyra Yee and Yann Dauphin and Michael Auli},
  booktitle = {Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}
```

## Pre-trained Models:

Model | Description |  Download
---|---|---
`transformer.noisychannel.de-en` | De->En Forward Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/forward_de2en.tar.bz2)
`transformer.noisychannel.en-de` | En->De Channel Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/backward_en2de.tar.bz2)
`transformer_lm.noisychannel.en` | En Language model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/reranking_en_lm.tar.bz2)

Test Data: [newstest_wmt17](https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/wmt17test.tar.bz2)

## Example usage

```
mkdir rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/forward_de2en.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/backward_en2de.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/reranking_en_lm.tar.bz2 | tar xvjf - -C rerank_example
curl https://dl.fbaipublicfiles.com/fairseq/models/noisychannel/wmt17test.tar.bz2 | tar xvjf - -C rerank_example

beam=50
num_trials=1000
fw_name=fw_model_ex
bw_name=bw_model_ex
lm_name=lm_ex
data_dir=rerank_example/hyphen-splitting-mixed-case-wmt17test-wmt14bpe
data_dir_name=wmt17
lm=rerank_example/lm/checkpoint_best.pt
lm_bpe_code=rerank_example/lm/bpe32k.code
lm_dict=rerank_example/lm/dict.txt
batch_size=32
bw=rerank_example/backward_en2de.pt
fw=rerank_example/forward_de2en.pt

# reranking with P(T|S) P(S|T) and P(T)
python examples/noisychannel/rerank_tune.py $data_dir  --tune-param lenpen weight1 weight3  \
    --lower-bound 0 0 0 --upper-bound 3 3 3 --data-dir-name $data_dir_name  \ 
    --num-trials $num_trials  --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model2 $fw --score-model1 $bw \
    --backwards1 --weight2 1 \
    -lm $lm  --lm-dict $lm_dict  --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code \
    --model2-name $fw_name --model1-name $bw_name --gen-model-name $fw_name

# reranking with P(T|S) and P(T)
python examples/noisychannel/rerank_tune.py $data_dir  --tune-param lenpen weight3 \
    --lower-bound 0 0 --upper-bound 3 3  --data-dir-name $data_dir_name  \
    --num-trials $num_trials  --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model1 $fw \
    -lm $lm  --lm-dict $lm_dict  --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code \
    --model1-name $fw_name --gen-model-name $fw_name

# to run with a preconfigured set of hyperparameters for the lenpen and model weights, using rerank.py instead.
python examples/noisychannel/rerank.py $data_dir \
    --lenpen 0.269 --weight1 1 --weight2 0.929 --weight3 0.831  \
    --data-dir-name $data_dir_name  --source-lang de --target-lang en --gen-model $fw \
    -n $beam --batch-size $batch_size --score-model2 $fw --score-model1 $bw --backwards1  \
    -lm $lm  --lm-dict $lm_dict  --lm-name en_newscrawl --lm-bpe-code $lm_bpe_code \
    --model2-name $fw_name --model1-name $bw_name --gen-model-name $fw_name
```

