FROM ubuntu:20.04

MAINTAINER Changhan Wang (changhan@fb.com)

RUN apt-get update && apt-get install -y build-essential git python3 python3-pip libsndfile1
RUN git clone -b simulastsharedtask https://github.com/pytorch/fairseq.git /fairseq
WORKDIR /fairseq

RUN pip3 install torch vizseq soundfile sentencepiece
RUN pip3 install -e .

RUN ln -s /usr/bin/python3 /usr/bin/python

ENTRYPOINT ./examples/simultaneous_translation/scripts/start-server.sh \
    examples/simultaneous_translation/scripts/configs/must-c-en_de-text-dev.sh & \
    (sleep 30 && ./examples/simultaneous_translation/scripts/start-multi-client.sh \
    examples/simultaneous_translation/scripts/configs/must-c-en_de-text-dev.sh \
    experiments/checkpoints/${CHKPT_FILENAME})
