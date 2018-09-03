.. role:: hidden
    :class: hidden-section

.. module:: fairseq.data

Data Loading and Utilities
==========================

.. _datasets:

Datasets
--------

**Datasets** define the data format and provide helpers for creating
mini-batches.

.. autoclass:: fairseq.data.FairseqDataset
    :members:
.. autoclass:: fairseq.data.LanguagePairDataset
    :members:
.. autoclass:: fairseq.data.MonolingualDataset
    :members:


Dictionary
----------

.. autoclass:: fairseq.data.Dictionary
    :members:


Iterators
---------

.. autoclass:: fairseq.data.CountingIterator
    :members:
.. autoclass:: fairseq.data.EpochBatchIterator
    :members:
.. autoclass:: fairseq.data.ShardedIterator
    :members:
