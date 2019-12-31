HOSTNAME=${1:-"localhost"}
PORT=${2:-"12321"}
SRC=${3:-"en"}
TGT=${4:-"de"}
PREFIX=$5

python ./eval/server.py \
    --hostname ${HOSTNAME} \
    --port ${PORT} \
    --debug \
    --src-file $PREFIX.$SRC \
    --ref-file $PREFIX.$TGT
