#! /usr/bin/bash
set -e

# set gpu device
export device=0,1,2,3,4,5,6,7

# task name
export task=wmt-en2de

# experiment name
export tag=deep-prenorm-dlcl

# save dir
export save_dir=checkpoints/$task/$tag
if [ ! -d $save_dir ]; then
	mkdir -p $save_dir
fi

# current running script
export script=${BASH_SOURCE[0]}

function main {

### training setting
# set fp16=1 if possible
fp16=
# model
arch=dlcl_transformer_prenorm_deep_wmt_en_de
# data set
data_dir=wmt16_en_de_google
# share embedding, only work for joined vocabulary
share_embedding=1
lr=0.002
warmup=16000
# token batch size per GPU
batch_size=4096
# accumulate gradient (equivalent to 4096 * 2 token batch size per GPU)
update_freq=2
weight_decay=0.0
# save last n checkpoint to save disk
saved_checkpoint_num=5
# training epoch
max_epoch=20
max_update=

### decoding setting
# report BLEU on validate set and test set
whos=(valid test)
# average last n checkpoints
ensemble=5
decoding_batch_size=32
beam=4
length_penalty=0.6

# copy training script
cp $script $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="python -u train.py data-bin/$data_dir
  --distributed-world-size $gpu_num -s en -t de
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens $batch_size
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 1000
  --save-dir $save_dir
  --save-last-checkpoints $saved_checkpoint_num"

adam_betas="'(0.9, 0.997)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$fp16" ]; then
cmd=${cmd}" --fp16 "
fi

export CUDA_VISIBLE_DEVICES=$device
log=$save_dir/train.log
if [ ! -e $log ]; then
	cmd="nohup "${cmd}" | tee $log &"
	eval $cmd
else
	for i in `seq 1 100`; do
		if [ ! -e $log.$i ]; then
			cmd="nohup "${cmd}" | tee $log.$i &"
			eval $cmd
			break
		fi
	done
fi
# wait for training finished
wait
echo -e " >> finish training \n"

if [ ! -e "$save_dir/last$ensemble.ensemble.pt" ]; then
	echo -e " >> generate last $ensemble ensemble model for inference \n"
	PYTHONPATH=`pwd` python scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
fi

checkpoint=last$ensemble.ensemble.pt

for ((i=0;i<${#whos[@]};i++));do
{
	# test set
	who=${whos[$i]}
	echo -e " >> translate $who by $checkpoint with batch=$batch_size beam=$beam alpha=$length_penalty\n"

	# translation log
	output=$save_dir/$who.trans.log

	# use the first gpu to decode
	gpu_id=$(echo $device | cut -d ',' -f 1)
	export CUDA_VISIBLE_DEVICES=$gpu_id

	python -u generate.py \
	data-bin/$data_dir \
	--path $save_dir/$checkpoint \
	--gen-subset $who \
	--batch-size $decoding_batch_size \
	--beam $beam \
	--lenpen $length_penalty \
	--log-format simple \
	--log-interval 10 \
	--remove-bpe > $output 2>&1

	echo -e " >> evaluate $who \n"
	bash ./scripts/wmt_en2de_google_multi_bleu.sh $output | tee -a $save_dir/train.log
}
done
}

export -f main
nohup bash -c main >> $save_dir/train.log 2>&1 &
sleep 2 && tail -f  $save_dir/train.log
wait
