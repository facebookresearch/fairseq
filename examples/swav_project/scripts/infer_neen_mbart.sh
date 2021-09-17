


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export src=ne && export tgt=en
export srcx=ne_NP && export tgtx=en_XX

export data="${data:-}"
export task="${task:-strtopfreqdy_augparascore_online_backtranslation_from_pretrained_bart}"

export ckpt="${ckpt:-}"

export lpst=${lpst:-1}
export lpts=${lpts:-1}
export inferst=${inferst:-1}
export inferts=${inferts:-1}

export beam=${beam:-5}
export tfreq=${tfreq:--1}

export maxtoks=${maxtoks:-4000}

if [ ! -f ${data} ]; then
    echo "data not found! ${data}"
    exit 1
fi

if [ ! -f ${ckpt} ]; then
    echo "ckpt not found! ${ckpt}"
    exit 1
fi


if [ ${prepend_bos} -eq 1 ]; then
    prepend_bos_s="--prepend-bos  "
else
    prepend_bos_s="  "
fi

echo "===================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "src: ${src} - ${tgt}; srcx: ${srcx}-${tgtx}"
echo "data: ${data}"
echo "task: ${task}"
echo "ckpt: ${ckpt}"
echo "lpst: ${lpst}"
echo "lpts: ${lpts}"
echo "inferst: ${inferst}"
echo "inferts: ${inferts}"

echo "beam: ${beam}"
echo "tfreq: ${tfreq}"
echo "maxtoks: ${maxtoks}"
echo "===================="


# data path
MAIN_PATH=$PWD
ROOT=$MAIN_PATH/$(dirname "$0")
TOOLS_PATH=${ROOT}/../../../../tools

FLORES_SCRIPTS=${TOOLS_PATH}/flores/floresv1/scripts

MULTIBLEU=${TOOLS_PATH}/mosesdecoder/scripts/generic/multi-bleu.perl
MOSES=${TOOLS_PATH}/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
# SCRIPTS=${FLORES_SCRIPTS}

if [ ! -f ${MULTIBLEU} ]; then
    echo "${MULTIBLEU} not found"
    exit 1
fi

if [ ! -f ${FLORES_SCRIPTS}/indic_norm_tok.sh ]; then
    echo "${FLORES_SCRIPTS}/indic_norm_tok.sh not found"
    exit 1
fi

SRC_TOKENIZER="bash $FLORES_SCRIPTS/indic_norm_tok.sh ${src}"


if [ ${inferst} -eq 1 ]; then
echo "===== ${srcx} --> ${tgtx} ====="
export fgen=$(mktemp /tmp/infer-fwd-script.XXXXXX)
echo "Saving in ${fgen}"
export fwd="fairseq-generate ${data} \
    --user-dir examples/swav_project/swav_src \
    --path ${ckpt} \
    --task ${task} \
    --top-frequency ${tfreq} \
    --gen-subset test \
    --mono-langs ${srcx},${tgtx} --valid-lang-pairs ${srcx}-${tgtx} \
    --remove-bpe 'sentencepiece' \
    --sacrebleu --scoring sacrebleu \
    --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
    --beam ${beam} 
    --lenpen ${lpst} \
    --max-tokens ${maxtoks} > ${fgen}
"
eval ${fwd}
tail -n 1 $fgen
cat $fgen | grep -P "^T-" | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape en > $fgen.ref
cat $fgen | grep -P "^H-" | cut -f3 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape en > $fgen.hyp
${MULTIBLEU} $fgen.ref < $fgen.hyp
fi


if [ ${inferts} -eq 1 ]; then
echo "===== ${tgtx} --> ${srcx} ====="
export bgen=$(mktemp /tmp/infer-script.XXXXXX)
echo "Saving in ${bgen}"
export bwd="fairseq-generate ${data} \
    --user-dir examples/swav_project/swav_src \
    --path ${ckpt} \
    --task ${task}  \
    --top-frequency ${tfreq} \
    --gen-subset test \
    --mono-langs ${srcx},${tgtx} --valid-lang-pairs ${tgtx}-${srcx} \
    --remove-bpe 'sentencepiece' \
    --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN \
    --beam ${beam} 
    --lenpen ${lpts} \
    --max-tokens ${maxtoks} > ${bgen}
"
eval ${bwd}
tail -n 1 $bgen
cat $bgen | grep -P "^T-" | cut -f2  > $bgen.ref
cat $bgen | grep -P "^H-" | cut -f3  > $bgen.hyp
$SRC_TOKENIZER $bgen.ref > $bgen.tok.ref
$SRC_TOKENIZER $bgen.hyp > $bgen.tok.hyp
${MULTIBLEU} $bgen.tok.ref < $bgen.tok.hyp
# rm -rf ${bgen}*
fi


fi




