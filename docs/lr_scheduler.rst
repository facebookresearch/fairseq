.. role:: hidden
    :class: hidden-section

.. _Learning Rate Schedulers:

Learning Rate Schedulers
========================

Learning Rate Schedulers update the learning rate over the course of training.
Learning rates can be updated after each update via :func:`step_update` or at
epoch boundaries via :func:`step`.

.. automodule:: fairseq.optim.lr_scheduler
    :members:

.. autoclass:: fairseq.optim.lr_scheduler.FairseqLRScheduler
    :members:
    :undoc-members:

.. autoclass:: fairseq.optim.lr_scheduler.cosine_lr_scheduler.CosineSchedule
    :members:
    :undoc-members:
.. autoclass:: fairseq.optim.lr_scheduler.fixed_schedule.FixedSchedule
    :members:
    :undoc-members:
.. autoclass:: fairseq.optim.lr_scheduler.inverse_square_root_schedule.InverseSquareRootSchedule
    :members:
    :undoc-members:
.. autoclass:: fairseq.optim.lr_scheduler.reduce_lr_on_plateau.ReduceLROnPlateau
    :members:
    :undoc-members:
.. autoclass:: fairseq.optim.lr_scheduler.reduce_angular_lr_scheduler.TriangularSchedule
    :members:
    :undoc-members:
