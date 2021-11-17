### Contrastive Clustering to Mine Pseudo-Parallel Data for Unsupervised Translation

ICLR Submission

### Introduction

`swav_lm` is a branch mainly implementing [SwAV](https://arxiv.org/pdf/2006.09882.pdf) (swapped prediction problem loss) and its variant (language agnostic constraint for SwAV loss (LangAgSwAV)) to apply Contrastive CLustering Assignments to NLP, text and language.

LangAgSwAV can be use to finetune pre-trained models so that they can become language-agnostic and semantic clustering. These properties may be necessary to mine pseudo-parallel data to augment Unsupervised MT

This project has 3 contribution:

* Language-Agnostic Constraint for SwAV loss (SwAV), which is used to finetune pre-trained models so that they can become language-agnostic and semantic clustering and can be used to mine data
* Mining Algorithm, which is combination of Margin-based mining and the **Filter Suite**
* Rank-based Cross Entropy and and dynamic lambda to integrate into UMT

### How this branch is organized

1. `examples/swav_project` mainly contain the instruction to use this branch, run files, experiments commands.
    1. `examples/swav_project/swav_src` has all the source codes for swav UMT
    2. `examples/swav_project/scripts` has functionality scripts that are necessary for the experiments


### Example runs:
#### Swav multilingual masked LM on pretrained MASS

```bash
# STEP 0 -- data preprocess
# For XLM model adapted from XLM codebase
#   use python examples/swav_project/scripts/xlm_preprocess.py
# 0.0: follow data preprocessing step from https://github.com/facebookresearch/XLM
#   must use the dictionary and bpe code provided by XLM paper
#   retrieve the tokenized monolingual data and valid and test data
# 0.1: convert the raw (tokenized) monolingual data into fairseq binary with
#   python examples/swav_project/scripts/xlm_preprocess.py
#       this script will build XLM custom dictionary, in which the order of <bos> and <eos> is swapped.
# 0.2: position the binary files into per-language folder:
#   data:
#       |-- en
#           |-- dict.txt  test.bin  test.idx  train.bin  train.idx  valid.bin  valid.idx
#       |-- ro
#           |-- dict.txt  test.bin  test.idx  train.bin  train.idx  valid.bin  valid.idx
#       |-- dict.en.txt  
#       |-- dict.ro.txt  
#       |-- dict.txt
#       |-- test.en-ro.en.bin  test.en-ro.en.idx  test.en-ro.ro.bin  test.en-ro.ro.idx	
#       |-- valid.en-ro.en.bin  valid.en-ro.en.idx	valid.en-ro.ro.bin  valid.en-ro.ro.idx
# 0.3: store it in variables $data

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export src=en
export tgt=ro


# STEP 1 -- finetune with language-agnostic swav loss
export ckptdir=
export ckptdir=<where the finetuned langag model is>
export ckptdir=/checkpoint/nxphi/testing/finetuned_swav_model

# NOTE: try to use as many physical GPUs as possible. Due to batch level SwAV loss computation, 
#   using update-freq=2 on 8 GPUs is NOT the same as 16 GPUs (distributed)
#   ideally, use 32 physical GPUs
fairseq-train ${data} \
    --user-dir examples/swav_project/swav_src \
    --save-dir ${ckptdir} \
    --rand-factor 8 \
    --swav-lambda 1.9 \
    --queue-length 8192 \
    --update-queue-starts 10000 \
    --stability-epsilon 1e-10 \
    --pre-norm-prototypes \
    --no-token-block --noising-module UnsupervisedMTNoisingNoBpe \
    --arch swav_xlm_encoder_big \
    --task multilingual_swav_lm_xlm \
    --criterion freq_langag_swav_masked_lm \
    --tokens-per-sample 512 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay --lr 0.0001 \
    --warmup-updates 10000 --total-num-update 125000 --max-update 20000 \
    --emb_dim 1024 --n_layers 6 --n_heads 8 --gelu_activation true --dropout 0.1 \
    --attention_dropout 0.1 --multilang-sampling-alpha 0.01 --max-tokens 3584 --update-freq 1 \
    --log-format json --log-interval 100 --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 2000 --keep-last-epochs 1 --fp16


# STEP 2 -- Export Swav Embeddings into disk
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export src=en
export tgt=ro
export data=/private/home/nxphi/train_data/fairseq-bin/big-pret-xlm-en-ro/lang_sep
export data=/private/home/nxphi/train_data/fairseq-bin/mass-pret-xlm-en-ro/lang_sep
export ckptdir=<where the finetuned langag model is>
# export ckptdir=/checkpoint/nxphi/2021-07-12/swav_langag_xlm_8_big_enro.rafa7.swlam0.5.q24576.qstart3k.eps10.prno.NoiNoBpe.xlm.swxlenbi.muswlmxl.langagc3preexpplus.drp0.1.eqratio.maxtoks3k.upfr1.ngpu24
export ckpt_name=checkpoint_best.pt
export ckpt=$ckptdir/$ckpt_name
export max_steps=50000

# eg devfair
export CUDA_VISIBLE_DEVICES=0,1
export max_steps=100

export swav_embed_path=$ckptdir/export_swav/${ckpt_name}.${max_steps}


rm -rf ${swav_embed_path}
export NGPU=$(echo ${CUDA_VISIBLE_DEVICES} | grep -o "," | wc -l)
python -m torch.distributed.launch --nproc_per_node=$NGPU examples/swav_project/scripts/export_swav_embeds.py \
    ${data} \
    --user-dir examples/swav_project/swav_src \
    --analyze-name swavembs \
    --analyze-max-step ${max_steps} \
    --dataset-impl mmap \
    --aly-exclude text,sinkhorn_prototypes,embed,tokens,lang_id \
    --export-flush-steps 5000 \
    --swav-langs ${src},${tgt} \
    --queue-length 0 \
    --path ${ckpt} \
    --no-token-block \
    --results-path ${swav_embed_path} \
    --task multilingual_swav_lm_xlm \
    --criterion swav_masked_lm \
    --swav-lambda 1.9 \
    --max-tokens 20000 \
    --skip-invalid-size-inputs-valid-test \
    --valid-subset train \
    --fp16

# save_path: ${swav_embed_path}/${lang}_part${i}/swavembs*.pth

# STEP 3: faiss mining NN search

export pth_name=swavembs
export txt_root=/private/home/nxphi/fixlm2/data/processed/pret-en-ro
export txt_prefix=train
export src_txt=${txt_root}/${txt_prefix}.${src}
export tgt_txt=${txt_root}/${txt_prefix}.${tgt}

for src_p in 0; do
    for tgt_p in 0; do
        echo "src_part ${src_p}, tgt_part ${tgt_p}"
        export src_part=${src}_part${src_p}
        export tgt_part=${tgt}_part${tgt_p}
        export out_part=${src}${tgt}_${src_p}${tgt_p}
        export src_pth=${swav_embed_path}/${src_part}/${pth_name}*${src}.r*.pth
        export tgt_pth=${swav_embed_path}/${tgt_part}/${pth_name}*${tgt}.r*.pth
        export out_pth=${swav_embed_path}/${out_part}/aligned.${pth_name}.idx.pth
        echo $src_pth
        python examples/swav_project/scripts/nn_search.py \
            --src-lang ${src} --tgt-lang ${tgt} \
            --src_pth ${src_pth} --tgt_pth ${tgt_pth} \
            --src_txt ${src_txt} --tgt_txt ${tgt_txt} \
            --output ${out_pth} \
            --mode mine --retrieval max --margin ratio -k 4 \
            --len_ratio 1.5 \
            --mem 15 \
            --verbose \
            --gpu 
    done
done

# expected output: ${swav_embed_path}/${out_part}/aligned.${pth_name}.idx.pth.txt.*

# STEP 4: aggregate mined data and filtering suite

export pct=0.05
export filter_minlen=5
export filter_maxlen=300
export ov_rate_max=0.35
export ov_mean_ratio=1.0
export filter_order=6
export prefix=aligned.swavembs.idx.pth
export outprefix=fil.${prefix}.pct${pct}.mlen${filter_minlen}.ovm${ov_rate_max}.omr${ov_mean_ratio}.o${filter_order}
unset src_txt tgt_txt index_file
for src_p in 0; do
    for tgt_p in 0; do
        export part_dir=${swav_embed_path}/${src}${tgt}_${src_p}${tgt_p}
        export src_txt=${src_txt},${part_dir}/${prefix}.txt.${src}
        export tgt_txt=${tgt_txt},${part_dir}/${prefix}.txt.${tgt}
        export index_file=${index_file},${part_dir}/${prefix}
    done
done

python examples/swav_project/scripts/para_txt_filter.py \
    --src-lang ${src} --tgt-lang ${tgt} \
    --src-txt ${src_txt} \
    --tgt-txt ${tgt_txt} \
    --index ${index_file} \
    --output ${swav_embed_path}/${outprefix} \
    --percentile ${pct} \
    --filter_minlen ${filter_minlen} \
    --filter_maxlen ${filter_maxlen} \
    --filter_overlap_rate_max ${ov_rate_max} \
    --filter_overlap_mean_ratio ${ov_mean_ratio} \
    --filter_same \
    --filter_order ${filter_order} \
    --log_examples 50 --bpe bpe \


# binarize the data

export prefix_dir=${swav_embed_path}
export dict=${data}/dict.${src}.txt

export destdir=${swav_embed_path}/${outprefix}/bin
export trainprefix=${swav_embed_path}/${outprefix}.pth.txt
export trainprefix=${swav_embed_path}/${outprefix}/export.txt

rm -rf ${destdir} && mkdir -p ${destdir} && cp -r ${swav_embed_path}/${outprefix}/index.pth ${destdir}/ && python examples/swav_project/scripts/xlm_preprocess.py \
  --source-lang ${src} \
  --target-lang ${tgt} \
  --trainpref ${trainprefix} \
  --destdir ${destdir} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${dict} \
  --tgtdict ${dict} \
  --xlm-mode \
  --workers 70 && \
for l in ${src} ${tgt}; do 
    for e in bin idx; do 
        echo "cp train.${src}-${tgt}.$l.$e"
        ln -s ${destdir}/train.${src}-${tgt}.${l}.${e} ${destdir}/valid.${src}-${tgt}.${l}.${e}  
    done
done && cp -r $dict ${destdir}/dict.txt

# binarized data saved: ${destdir}


# STEP 5: train UMT models with augmentation data

export save_dir=<specify the save dir>
export pretrained_model=mass_enro.pt

export save_dir=/checkpoint/nxphi/testing/umtout_${src}${tgt}
export pretrained_model=/private/home/nxphi/train_out/mass-pret-xlm-en-ro-test-train/exp_mass_mt_pretrained_from_xlm/checkpoint_pret_mass.pt

fairseq-train ${data} \
    --user-dir examples/swav_project/swav_src \
    --save-dir ${save_dir} --lambda-dae 0 \
    --augpara-path ${destdir} \
    --augpara-pairs ${src}-${tgt} \
    --dataset-impl mmap \
    --lambda-augpara '0:1,5000:0.2,10000:0.15' \
    --augpara-reverse \
    --scores2weights uniform_rank \
    --scores2weights-params 0.2,1.0 \
    --finetune-from-model ${pretrained_model} \
    --arch mass_transformer_big \
    --task umt_augpara_score_online_backtranslation_xlm \
    --criterion weighted_label_smoothed_cross_entropy \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-06 --lr 0.0001 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --clip-norm 5.0 --max-update 125000 \
    --emb_dim 1024 --n_layers 6 --n_heads 8 --gelu_activation true \
    --attention_setting v1 --dropout 0.1 --n_langs 2 \
    --attention_dropout 0.1 --mono_langs_no_ignore true --mono-langs ${src},${tgt} \
    --valid-lang-pairs ${src}-${tgt},${tgt}-${src} \
    --eval-bleu --eval-tokenized-bleu \
    --eval-bleu-remove-bpe --eval-bleu-args '{"beam": 5, "lenpen": 1}' \
    --eval-bleu-detok-args '{}' \
    --max-tokens 4096 --update-freq 1 --log-format json --log-interval 100 \
    --skip-invalid-size-inputs-valid-test --save-interval-updates 1000 \
    --keep-last-epochs 1 --eval-bleu-bwd --valid-subset valid \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --fp16


# STEP 6: inference MT model
#   may need to tune --lenpen

export save_dir=/checkpoint/nxphi/testing/umtout_${src}${tgt}
export ckpt=${save_dir}/checkpoint_best.pt

# src->tgt
fairseq-generate ${data} \
    --user-dir examples/swav_project/swav_src \
    --dataset-impl mmap \
    --path ${ckpt} \
    --gen-subset test \
    --task umt_augpara_score_online_backtranslation_xlm \
    --mono-langs ${src},${tgt} --valid-lang-pairs ${src}-${tgt} \
    --scoring bleu \
    --remove-bpe --max-tokens 6000 --beam 5 --lenpen 0.6 --quiet

# tgt->src
fairseq-generate ${data} \
    --user-dir examples/swav_project/swav_src \
    --dataset-impl mmap \
    --path ${ckpt} \
    --gen-subset test \
    --task umt_augpara_score_online_backtranslation_xlm \
    --mono-langs ${src},${tgt} --valid-lang-pairs ${tgt}-${src} \
    --scoring bleu \
    --remove-bpe --max-tokens 6000 --beam 5 --lenpen 0.6 --quiet


```



