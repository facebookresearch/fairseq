output_dir=$1
drop=$2
cmr_gate_drop=$3
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.resume_finished=false \
    cfg.output_dir=$output_dir \
    cfg.max_updates=100000 \
    cfg.keep_interval_updates=3 \
    cfg.save_interval=1000 \
    cfg.symlink_best_and_last_checkpoints=true \
    cfg.dataset.eval_lang_pairs_file=examples/nllb/modeling/scripts/flores200/eval_lang_pairs_eng400_noneng20.txt \
    cfg.dropout=${drop} \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq 2 --moe-cmr --cmr-gate-drop $cmr_gate_drop " \
-c job
