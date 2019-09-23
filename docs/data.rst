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

**Helper Datasets**

These datasets wrap other :class:`fairseq.data.FairseqDataset` instances and
provide additional functionality:

.. autoclass:: fairseq.data.BacktranslationDataset
    :members:
.. autoclass:: fairseq.data.ConcatDataset
    :members:
.. autoclass:: fairseq.data.ResamplingDataset
    :members:
.. autoclass:: fairseq.data.RoundRobinZipDatasets
    :members:
.. autoclass:: fairseq.data.TransformEosDataset
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
.. autoclass:: fairseq.data.GroupedIterator
    :members:
.. autoclass:: fairseq.data.ShardedIterator
    :members:
