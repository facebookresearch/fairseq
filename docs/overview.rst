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

where the default implementation for ``train.train_step`` is roughly::

  def train_step(self, batch, model, criterion, optimizer):
      loss = criterion(model, batch)
      optimizer.backward(loss)

**Registering new plug-ins**

New plug-ins are *registered* through a set of ``@register`` function
decorators, for example::

  @register_model('my_lstm')
  class MyLSTM(FairseqModel):
      (...)

Once registered, new plug-ins can be used with the existing :ref:`Command-line
Tools`. See the Tutorial sections for more detailed walkthroughs of how to add
new plug-ins.
