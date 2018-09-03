Evaluating Pre-trained Models
=============================

First, download a pre-trained model along with its vocabularies:

.. code-block:: console

    > curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -

This model uses a `Byte Pair Encoding (BPE)
vocabulary <https://arxiv.org/abs/1508.07909>`__, so we'll have to apply
the encoding to the source text before it can be translated. This can be
done with the
`apply\_bpe.py <https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py>`__
script using the ``wmt14.en-fr.fconv-cuda/bpecodes`` file. ``@@`` is
used as a continuation marker and the original text can be easily
recovered with e.g. ``sed s/@@ //g`` or by passing the ``--remove-bpe``
flag to :ref:`generate.py`. Prior to BPE, input text needs to be tokenized
using ``tokenizer.perl`` from
`mosesdecoder <https://github.com/moses-smt/mosesdecoder>`__.

Let's use :ref:`interactive.py` to generate translations
interactively. Here, we use a beam size of 5:

.. code-block:: console

    > MODEL_DIR=wmt14.en-fr.fconv-py
    > python interactive.py \
        --path $MODEL_DIR/model.pt $MODEL_DIR \
        --beam 5
    | loading model(s) from wmt14.en-fr.fconv-py/model.pt
    | [en] dictionary: 44206 types
    | [fr] dictionary: 44463 types
    | Type the input sentence and press return:
    > Why is it rare to discover new marine mam@@ mal species ?
    O       Why is it rare to discover new marine mam@@ mal species ?
    H       -0.06429661810398102    Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
    A       0 1 3 3 5 6 6 8 8 8 7 11 12

This generation script produces four types of outputs: a line prefixed
with *S* shows the supplied source sentence after applying the
vocabulary; *O* is a copy of the original source sentence; *H* is the
hypothesis along with an average log-likelihood; and *A* is the
attention maxima for each word in the hypothesis, including the
end-of-sentence marker which is omitted from the text.

See the `README <https://github.com/pytorch/fairseq#pre-trained-models>`__ for a
full list of pre-trained models available.

Training a New Model
====================

The following tutorial is for machine translation. For an example of how
to use Fairseq for other tasks, such as :ref:`language modeling`, please see the
``examples/`` directory.

Data Pre-processing
-------------------

Fairseq contains example pre-processing scripts for several translation
datasets: IWSLT 2014 (German-English), WMT 2014 (English-French) and WMT
2014 (English-German). To pre-process and binarize the IWSLT dataset:

.. code-block:: console

    > cd examples/translation/
    > bash prepare-iwslt14.sh
    > cd ../..
    > TEXT=examples/translation/iwslt14.tokenized.de-en
    > python preprocess.py --source-lang de --target-lang en \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/iwslt14.tokenized.de-en

This will write binarized data that can be used for model training to
``data-bin/iwslt14.tokenized.de-en``.

Training
--------

Use :ref:`train.py` to train a new model. Here a few example settings that work
well for the IWSLT 2014 dataset:

.. code-block:: console

    > mkdir -p checkpoints/fconv
    > CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
        --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
        --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

By default, :ref:`train.py` will use all available GPUs on your machine. Use the
``CUDA_VISIBLE_DEVICES`` environment variable to select specific GPUs and/or to
change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum
number of tokens per batch (``--max-tokens``). You may need to use a
smaller value depending on the available GPU memory on your system.

Generation
----------

Once your model is trained, you can generate translations using
:ref:`generate.py` **(for binarized data)** or
:ref:`interactive.py` **(for raw text)**:

.. code-block:: console

    > python generate.py data-bin/iwslt14.tokenized.de-en \
        --path checkpoints/fconv/checkpoint_best.pt \
        --batch-size 128 --beam 5
    | [de] dictionary: 35475 types
    | [en] dictionary: 24739 types
    | data-bin/iwslt14.tokenized.de-en test 6750 examples
    | model fconv
    | loaded checkpoint trainings/fconv/checkpoint_best.pt
    S-721   danke .
    T-721   thank you .
    ...

To generate translations with only a CPU, use the ``--cpu`` flag. BPE
continuation markers can be removed with the ``--remove-bpe`` flag.

Advanced Training Options
=========================

Large mini-batch training with delayed updates
----------------------------------------------

The ``--update-freq`` option can be used to accumulate gradients from
multiple mini-batches and delay updating, creating a larger effective
batch size. Delayed updates can also improve training speed by reducing
inter-GPU communication costs and by saving idle time caused by variance
in workload across GPUs. See `Ott et al.
(2018) <https://arxiv.org/abs/1806.00187>`__ for more details.

To train on a single GPU with an effective batch size that is equivalent
to training on 8 GPUs:

.. code-block:: console

    > CUDA_VISIBLE_DEVICES=0 python train.py --update-freq 8 (...)

Training with half precision floating point (FP16)
--------------------------------------------------

.. note::

    FP16 training requires a Volta GPU and CUDA 9.1 or greater

Recent GPUs enable efficient half precision floating point computation,
e.g., using `Nvidia Tensor Cores
<https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html>`__.
Fairseq supports FP16 training with the ``--fp16`` flag:

.. code-block:: console

    > python train.py --fp16 (...)

Distributed training
--------------------

Distributed training in fairseq is implemented on top of
`torch.distributed <http://pytorch.org/docs/master/distributed.html>`__.
Training begins by launching one worker process per GPU. These workers
discover each other via a unique host and port (required) that can be
used to establish an initial connection. Additionally, each worker has a
rank, that is a unique number from 0 to n-1 where n is the total number
of GPUs.

If you run on a cluster managed by
`SLURM <https://slurm.schedmd.com/>`__ you can train a large
English-French model on the WMT 2014 dataset on 16 nodes with 8 GPUs
each (in total 128 GPUs) using this command:

.. code-block:: console

    > DATA=...   # path to the preprocessed dataset, must be visible from all nodes
    > PORT=9218  # any available TCP port that can be used by the trainer to establish initial connection
    > sbatch --job-name fairseq-py --gres gpu:8 --cpus-per-task 10 \
        --nodes 16 --ntasks-per-node 8 \
        --wrap 'srun --output train.log.node%t --error train.stderr.node%t.%j \
        python train.py $DATA \
        --distributed-world-size 128 \
        --distributed-port $PORT \
        --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
        --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
        --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 --wd 0.0001'

Alternatively you can manually start one process per GPU:

.. code-block:: console

    > DATA=...  # path to the preprocessed dataset, must be visible from all nodes
    > HOST_PORT=master.example.com:9218  # one of the hosts used by the job
    > RANK=...  # the rank of this process, from 0 to 127 in case of 128 GPUs
    > python train.py $DATA \
        --distributed-world-size 128 \
        --distributed-init-method 'tcp://$HOST_PORT' \
        --distributed-rank $RANK \
        --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
        --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
        --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 --wd 0.0001
