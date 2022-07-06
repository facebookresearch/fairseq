#!/bin/bash

# Script for generating backtranslations.
# Usage `examples/nllb/modeling/scripts/generate_backtranslations.sh (x_en|en_x) (lang1,lang2,lang3...)`
# Monoglingual data located in ${MONODIR}. Generated output BT data will be in ${OUTDIR}.
# search for decoding via $1.

MODEL=/path/to/model/checkpoint/
MODEL_LANGS="ace_Latn,afr_Latn,arb_Arab,ast_Latn,ayr_Latn,bel_Cyrl,bul_Cyrl,cjk_Latn,cym_Latn,eng_Latn,eus_Latn,ewe_Latn,pes_Arab,fin_Latn,fon_Latn,fra_Latn,fuv_Latn,hau_Latn,hin_Deva,isl_Latn,ita_Latn,jpn_Jpan,kea_Latn,kik_Latn,kin_Latn,kon_Latn,kor_Hang,lvs_Latn,lin_Latn,luo_Latn,mal_Mlym,mar_Deva,nso_Latn,oci_Latn,por_Latn,run_Latn,rus_Cyrl,sin_Sinh,snd_Arab,swh_Latn,tam_Taml,tat_Cyrl,tel_Telu,tir_Ethi,tsn_Latn,tso_Latn,twi_Latn,urd_Arab,vie_Latn,wol_Latn,yor_Latn,yue_Hant,zho_Hans"

MONODIR=/path/to/monolingual/data
OUTDIR=/path/to/output/generated/data
mkdir -p ${OUTDIR}

PARTITION=nllb
MAX_TOKENS=2560
GPUS_PER_NODE=1
NODES=1
CPUS_PER_TASK=8
TIME=2000
MEM=200G
DECODE_ARGS="--beam 4"

if [[ $# -ne 2 ]] ; then
  echo "syntax: ${0} (x_en|en_x) (lang1,lang2,lang3...)"
  exit 1
fi


DIRECTION=$1
LANGS=$2


# Gets a list of shards to generate, excluding those that have already been completed.
get_incomplete_shards() {
  num_shards=$1
  src=$2
  tgt=$3
  array=
  for shard in $(seq 0 $((${num_shards} - 1))); do
    # If we don't have any logs for this shard, queue the job
    shard_log_count=$(ls ${OUTDIR}/${src}-${tgt}/*_${shard}.out 2> /dev/null | head -c1 | wc -c)
    if [ $shard_log_count -eq 0 ]; then
      array=${array}${shard},
      continue
    fi

    padded_shard=$(printf %03d ${shard})
    idx=${MONODIR}/data_bin/shard${padded_shard}/train.${src}-${src}.${src}.idx
    idx_num_lines=$(python examples/nllb/modeling/scripts/backtranslation/count_idx.py ${idx})
    # Use the first complete output log, delete the others
    found_complete_shard=false
    for output in ${OUTDIR}/${src}-${tgt}/*_${shard}.out; do
      out_num_lines=$(cat ${output} | grep "^H-" | wc -l)
      if [ "${found_complete_shard}" = true ]; then
        rm -f ${output}
        rm -f ${f/.out/.err}
      elif [ ${out_num_lines} -gt $(echo "(${idx_num_lines}*0.95)/1" | bc) ]; then
        found_complete_shard=true
      else
        rm -f ${output}
        rm -f ${output/.out/.err}
      fi

    done

    if [ "${found_complete_shard}" = false ]; then
      array=${array}${shard},
    fi

  done
  if [ ! -z "$array" ]; then
    array=${array%?}
  fi
  echo $array
}


########
# X-EN #
########


if [ "${DIRECTION}" = "x_en" ]; then
  for lang in ${LANGS//,/ }; do
    if [ ! -f ${MONODIR}/data_bin/shard000/dict.${lang}.txt ]; then
      echo Missing data for ${lang}, skipping.
      continue
    fi
    mkdir -p ${OUTDIR}/${lang}-eng

    num_shards=$(ls -1 ${MONODIR}/data_bin/shard*/train.${lang}-${lang}.${lang}.idx | wc -l)
    shards_to_run="$(get_incomplete_shards ${num_shards} ${lang} eng)"
    if [ ! -z "${shards_to_run}" ]; then
      echo Kicking off backtranslation of ${lang}-eng for shards ${shards_to_run}
      echo "shard=\$(printf %03d \${SLURM_ARRAY_TASK_ID})
      shard_dir=${MONODIR}/data_bin/shard\$shard
      if [ ! -f \$shard_dir/train.${lang}-eng.${lang}.bin ]; then
        ln -s \$shard_dir/train.${lang}-${lang}.${lang}.bin \$shard_dir/train.${lang}-eng.${lang}.bin
      fi
      if [ ! -f \$shard_dir/train.${lang}-eng.${lang}.idx ]; then
        ln -s \$shard_dir/train.${lang}-${lang}.${lang}.idx \$shard_dir/train.${lang}-eng.${lang}.idx
      fi
      python fairseq_cli/generate.py \$shard_dir \
        --path ${MODEL} \
        --task=translation_multi_simple_epoch \
        --langs ${MODEL_LANGS} \
        --lang-pairs ${lang}-eng \
        --source-lang ${lang} --target-lang eng \
        --encoder-langtok "src" --decoder-langtok \
        --add-data-source-prefix-tags \
        --gen-subset train \
        --max-tokens ${MAX_TOKENS} \
        --skip-invalid-size-inputs-valid-test \
        ${DECODE_ARGS}" > ${OUTDIR}/${lang}-eng.job

      sbatch --output ${OUTDIR}/${lang}-eng/%A_%a.out \
        --error ${OUTDIR}/${lang}-eng/%A_%a.err \
        --job-name bt.${lang}-eng --array=${shards_to_run} \
        --gpus-per-node ${GPUS_PER_NODE} --nodes ${NODES} --cpus-per-task ${CPUS_PER_TASK} \
        --time ${TIME} --mem $MEM -C volta32gb --partition ${PARTITION} \
        --ntasks-per-node 1 --open-mode append --no-requeue \
        --wrap "srun sh ${OUTDIR}/${lang}-eng.job"
    else
      echo No shards left to do for ${lang}-eng
    fi
  done
fi


########
# EN-X #
########


if [ "${DIRECTION}" = "en_x" ]; then
  num_shards=$(ls -1 ${MONODIR}/data_bin/shard*/train.eng-eng.eng.idx | wc -l)
  for lang in ${LANGS//,/ }; do
    if [ ! -f ${MONODIR}/data_bin/shard000/dict.eng.txt ]; then
      echo Missing data for eng, skipping.
      continue
    fi
    mkdir -p ${OUTDIR}/eng-${lang}

    shards_to_run="$(get_incomplete_shards ${num_shards} eng ${lang})"
    if [ ! -z "${shards_to_run}" ]; then
      echo Kicking off backtranslation of eng-${lang} for shards ${shards_to_run}

      echo "shard=\$(printf %03d \${SLURM_ARRAY_TASK_ID})
      shard_dir=${MONODIR}/data_bin/shard\$shard
      if [ ! -f \$shard_dir/train.eng-${lang}.eng.bin ]; then
        ln -s \$shard_dir/train.eng-eng.eng.bin \$shard_dir/train.eng-${lang}.eng.bin
      fi
      if [ ! -f \$shard_dir/train.eng-${lang}.eng.idx ]; then
        ln -s \$shard_dir/train.eng-eng.eng.idx \$shard_dir/train.eng-${lang}.eng.idx
      fi
      python fairseq_cli/generate.py \$shard_dir \
        --fp16 \
        --path ${MODEL} \
        --task=translation_multi_simple_epoch \
        --langs ${MODEL_LANGS} \
        --lang-pairs eng-${lang} \
        --source-lang eng --target-lang ${lang} \
        --encoder-langtok "src" --decoder-langtok \
        --add-data-source-prefix-tags \
        --gen-subset train \
        --max-tokens ${MAX_TOKENS} \
        --skip-invalid-size-inputs-valid-test \
        ${DECODE_ARGS}" > ${OUTDIR}/eng-${lang}.job

      sbatch --output ${OUTDIR}/eng-${lang}/%A_%a.out \
        --error ${OUTDIR}/eng-${lang}/%A_%a.err \
        --job-name bt.eng-${lang} --array=${shards_to_run} \
        --gpus-per-node ${GPUS_PER_NODE} --nodes ${NODES} --cpus-per-task ${CPUS_PER_TASK} \
        --time ${TIME} --mem $MEM -C volta32gb --partition ${PARTITION} \
        --ntasks-per-node 1 --open-mode append --no-requeue \
        --wrap "srun sh ${OUTDIR}/eng-${lang}.job"
    else
      echo No shards left to do for eng-${lang}
    fi
  done
fi
