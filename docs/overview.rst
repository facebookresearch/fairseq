Overview
========

Fairseq can be extended through user-supplied `plug-ins
<https://en.wikipedia.org/wiki/Plug-in_(computing)>`_. We support five kinds of
plug-ins:

- :ref:`Models` define the neural network architecture and encapsulate all of the
  learnable parameters.
- :ref:`Criterions` compute the loss function given the model outputs and targets.
- :ref:`Tasks` store dictionaries and provide helpers for loading/iterating over
  Datasets, initializing the Model/Criterion and calculating the loss.
- :ref:`Optimizers` update the Model parameters based on the gradients.
- :ref:`Learning Rate Schedulers` update the learning rate over the course of
  training.

**Training Flow**

Given a ``model``, ``criterion``, ``task``, ``optimizer`` and ``lr_scheduler``,
fairseq implements the following high-level training flow::

  for epoch in range(num_epochs):
      itr = task.get_batch_iterator(task.dataset('train'))
      for num_updates, batch in enumerate(itr):
          task.train_step(batch, model, criterion, optimizer)
          average_and_clip_gradients()
          optimizer.step()
          lr_scheduler.step_update(num_updates)
      lr_scheduler.step(epoch)

where the default implementation for ``task.train_step`` is roughly::

  def train_step(self, batch, model, criterion, optimizer, **unused):
      loss = criterion(model, batch)
      optimizer.backward(loss)
      return loss

**Registering new plug-ins**

New plug-ins are *registered* through a set of ``@register`` function
decorators, for example::

  @register_model('my_lstm')
  class MyLSTM(FairseqEncoderDecoderModel):
      (...)

Once registered, new plug-ins can be used with the existing :ref:`Command-line
Tools`. See the Tutorial sections for more detailed walkthroughs of how to add
new plug-ins.

**Loading plug-ins from another directory**

New plug-ins can be defined in a custom module stored in the user system. In
order to import the module, and make the plugin available to *fairseq*, the
command line supports the ``--user-dir`` flag that can be used to specify a
custom location for additional modules to load into *fairseq*.

For example, assuming this directory tree::

  /home/user/my-module/
  └── __init__.py
  
with ``__init__.py``::

  from fairseq.models import register_model_architecture
  from fairseq.models.transformer import transformer_vaswani_wmt_en_de_big

  @register_model_architecture('transformer', 'my_transformer')
  def transformer_mmt_big(args):
      transformer_vaswani_wmt_en_de_big(args)

it is possible to invoke the :ref:`fairseq-train` script with the new architecture with::

  fairseq-train ... --user-dir /home/user/my-module -a my_transformer --task translation
