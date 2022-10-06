output_dir=$1
drop=$2
moe_eom=$3
moe_freq=2
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$output_dir \
    cfg.resume_finished=true \
    cfg.max_update_str="mu.200.300" \
    cfg.max_updates=200000 \
    cfg.dataset.lang_pairs_file=examples/nllb/modeling/scripts/flores200/cl1_lang_pairs.txt \
    cfg.dropout=${drop} \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq $moe_freq --moe-eom $moe_eom " \
-c job
