FROM ubuntu:20.04

MAINTAINER Changhan Wang (changhan@fb.com)

RUN apt-get update && apt-get install -y build-essential git python3 python3-pip libsndfile1
RUN git clone -b simulastsharedtask https://github.com/pytorch/fairseq.git /fairseq
WORKDIR /fairseq

RUN pip3 install torch vizseq soundfile sentencepiece sacrebleu=="1.4.5"
RUN pip3 install -e .

RUN ln -s /usr/bin/python3 /usr/bin/python

ENTRYPOINT python examples/simultaneous_translation/eval/server.py \
    --tokenizer 13a \
    --src-file $SRC_FILE\
    --tgt-file $TGT_FILE \
    --scorer-type text \
    --port $PORT \
    --debug \
    & (sleep 30 && eval $CLIENT_COMMAND)
