VOCAB_SIZE=12
SEQ_LEN=5

echo $LOAD

INIT_MDL=$LOAD/checkpoint100.pt

RUN="./scripts/run.sh"

EXP_NAME=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')

mkdir -p /checkpoint/llajan/$EXP_NAME

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -f -l $LAYERS"

if [ $FINETUNE == "no" ]; then
    ARGS="$ARGS --no-training"
elif [ $FINETUNE == "z" ]; then
    ARGS="$ARGS -z --lr 1e-2 --zsize $ZSIZE --task-emb-init zeros"
fi

if [[ $MDL == "snail" || $MDL == "matching" ]]; then
	ARGS="$ARGS --meta-ex $((TEST_SAMPLES+1))"
fi

if [ ! -z $INIT_MDL ]; then
    ARGS="$ARGS -i $INIT_MDL"
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

$RUN $ARGS --test-samples $TEST_SAMPLES -t 0 --eval

rm /checkpoint/llajan/$EXP_NAME -rf
