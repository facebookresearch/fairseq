VOCAB_SIZE=12
SEQ_LEN=5

RUN="./scripts/run.sh"

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 100"

if [[ $MDL == "snail" || $MDL == "matching" ]]; then
	  ARGS="$ARGS --meta-ex $((TEST_SAMPLES+1))"
fi

if [[ $MODE == *"meta"* ]]; then
    ARGS="$ARGS --zlr 1e-2 --zsize $ZSIZE --numgrads $NUMGRADS"
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
