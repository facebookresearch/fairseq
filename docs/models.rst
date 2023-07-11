.. role:: hidden
    :class: hidden-section

.. module:: fairseq.models

.. _Models:

Models
======

A Model defines the neural network's ``forward()`` method and encapsulates all
of the learnable parameters in the network. Each model also provides a set of
named *architectures* that define the precise network configuration (e.g.,
embedding dimension, number of layers, etc.).

Both the model type and architecture are selected via the ``--arch``
command-line argument. Once selected, a model may expose additional command-line
arguments for further configuration.

.. note::

    All fairseq Models extend :class:`BaseFairseqModel`, which in turn extends
    :class:`torch.nn.Module`. Thus any fairseq Model can be used as a
    stand-alone Module in other PyTorch code.


Convolutional Neural Networks (CNN)
-----------------------------------

.. module:: fairseq.models.fconv
.. autoclass:: fairseq.models.fconv.FConvModel
    :members:
.. autoclass:: fairseq.models.fconv.FConvEncoder
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.fconv.FConvDecoder
    :members:


Long Short-Term Memory (LSTM) networks
--------------------------------------

.. module:: fairseq.models.lstm
.. autoclass:: fairseq.models.lstm.LSTMModel
    :members:
.. autoclass:: fairseq.models.lstm.LSTMEncoder
    :members:
.. autoclass:: fairseq.models.lstm.LSTMDecoder
    :members:


Transformer (self-attention) networks
-------------------------------------

.. module:: fairseq.models.transformer
.. autoclass:: fairseq.models.transformer.TransformerModel
    :members:
.. autoclass:: fairseq.models.transformer.TransformerEncoder
    :members:
.. autoclass:: fairseq.models.transformer.TransformerEncoderLayer
    :members:
.. autoclass:: fairseq.models.transformer.TransformerDecoder
    :members:
.. autoclass:: fairseq.models.transformer.TransformerDecoderLayer
    :members:


Adding new models
-----------------

.. currentmodule:: fairseq.models
.. autofunction:: fairseq.models.register_model
.. autofunction:: fairseq.models.register_model_architecture
.. autoclass:: fairseq.models.BaseFairseqModel
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.FairseqEncoderDecoderModel
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.FairseqEncoderModel
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.FairseqLanguageModel
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.FairseqMultiModel
    :members:
    :undoc-members:
.. autoclass:: fairseq.models.FairseqEncoder
    :members:
.. autoclass:: fairseq.models.CompositeEncoder
    :members:
.. autoclass:: fairseq.models.FairseqDecoder
    :members:


.. _Incremental decoding:

Incremental decoding
--------------------

.. autoclass:: fairseq.models.FairseqIncrementalDecoder
    :members:
    :undoc-members:
