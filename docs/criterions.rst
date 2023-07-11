.. role:: hidden
    :class: hidden-section

.. _Criterions:

Criterions
==========

Criterions compute the loss function given the model and batch, roughly::

  loss = criterion(model, batch)

.. automodule:: fairseq.criterions
    :members:

.. autoclass:: fairseq.criterions.FairseqCriterion
    :members:
    :undoc-members:

.. autoclass:: fairseq.criterions.adaptive_loss.AdaptiveLoss
    :members:
    :undoc-members:
.. autoclass:: fairseq.criterions.composite_loss.CompositeLoss
    :members:
    :undoc-members:
.. autoclass:: fairseq.criterions.cross_entropy.CrossEntropyCriterion
    :members:
    :undoc-members:
.. autoclass:: fairseq.criterions.label_smoothed_cross_entropy.LabelSmoothedCrossEntropyCriterion
    :members:
    :undoc-members:
