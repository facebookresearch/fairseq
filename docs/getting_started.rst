Evaluating Pre-trained Models
=============================

First, download a pre-trained model along with its vocabularies:

.. code-block:: console

    > curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -

This model uses a `Byte Pair Encoding (BPE)
vocabulary <https://arxiv.org/abs/1508.07909>`__, so we'll have to apply
the encoding to the source text before it can be translated. This can be
done with the
`apply\_bpe.py <https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py>`__
script using the ``wmt14.en-fr.fconv-cuda/bpecodes`` file. ``@@`` is
used as a continuation marker and the original text can be easily
recovered with e.g. ``sed s/@@ //g`` or by passing the ``--remove-bpe``
flag to :ref:`fairseq-generate`. Prior to BPE, input text needs to be tokenized
using ``tokenizer.perl`` from
`mosesdecoder <https://github.com/moses-smt/mosesdecoder>`__.

Let's use :ref:`fairseq-interactive` to generate translations
interactively. Here, we use a beam size of 5:

.. code-block:: console

    > MODEL_DIR=wmt14.en-fr.fconv-py
    > fairseq-interactive \
        --path $MODEL_DIR/model.pt $MODEL_DIR \
        --beam 5 --source-lang en --target-lang fr
    | loading model(s) from wmt14.en-fr.fconv-py/model.pt
    | [en] dictionary: 44206 types
    | [fr] dictionary: 44463 types
    | Type the input sentence and press return:
    > Why is it rare to discover new marine mam@@ mal species ?
    O       Why is it rare to discover new marine mam@@ mal species ?
    H       -0.1525060087442398     Pourquoi est @-@ il rare de découvrir de nouvelles espèces de mammifères marins ?
    P       -0.2221 -0.3122 -0.1289 -0.2673 -0.1711 -0.1930 -0.1101 -0.1660 -0.1003 -0.0740 -0.1101 -0.0814 -0.1238 -0.0985 -0.1288

This generation script produces three types of outputs: a line prefixed
with *O* is a copy of the original source sentence; *H* is the
hypothesis along with an average log-likelihood; and *P* is the
positional score per token position, including the
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
    > fairseq-preprocess --source-lang de --target-lang en \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/iwslt14.tokenized.de-en

This will write binarized data that can be used for model training to
``data-bin/iwslt14.tokenized.de-en``.

Training
--------

Use :ref:`fairseq-train` to train a new model. Here a few example settings that work
well for the IWSLT 2014 dataset:

.. code-block:: console

    > mkdir -p checkpoints/fconv
    > CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
        --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
        --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

By default, :ref:`fairseq-train` will use all available GPUs on your machine. Use the
``CUDA_VISIBLE_DEVICES`` environment variable to select specific GPUs and/or to
change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum
number of tokens per batch (``--max-tokens``). You may need to use a
smaller value depending on the available GPU memory on your system.

Generation
----------

Once your model is trained, you can generate translations using
:ref:`fairseq-generate` **(for binarized data)** or
:ref:`fairseq-interactive` **(for raw text)**:

.. code-block:: console

    > fairseq-generate data-bin/iwslt14.tokenized.de-en \
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

    > CUDA_VISIBLE_DEVICES=0 fairseq-train --update-freq 8 (...)

Training with half precision floating point (FP16)
--------------------------------------------------

.. note::

    FP16 training requires a Volta GPU and CUDA 9.1 or greater

Recent GPUs enable efficient half precision floating point computation,
e.g., using `Nvidia Tensor Cores
<https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html>`__.
Fairseq supports FP16 training with the ``--fp16`` flag:

.. code-block:: console

    > fairseq-train --fp16 (...)

Lazily loading large training datasets
--------------------------------------

By default fairseq loads the entire training set into system memory. For large
datasets, the ``--lazy-load`` option can be used to instead load batches on-demand.
For optimal performance, use the ``--num-workers`` option to control the number
of background processes that will load batches.

Distributed training
--------------------

Distributed training in fairseq is implemented on top of ``torch.distributed``.
The easiest way to launch jobs is with the `torch.distributed.launch
<https://pytorch.org/docs/stable/distributed.html#launch-utility>`__ tool.

For example, to train a large English-German Transformer model on 2 nodes each
with 8 GPUs (in total 16 GPUs), run the following command on each node,
replacing ``node_rank=0`` with ``node_rank=1`` on the second node:

.. code-block:: console

    > python -m torch.distributed.launch --nproc_per_node=8 \
        --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" \
        --master_port=1234 \
        $(which fairseq-train) data-bin/wmt16_en_de_bpe32k \
        --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --lr 0.0005 --min-lr 1e-09 \
        --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 3584 \
        --fp16
