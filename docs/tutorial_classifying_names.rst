Tutorial: Classifying Names with a Character-Level RNN
======================================================

In this tutorial we will extend fairseq to support *classification* tasks. In
particular we will re-implement the PyTorch tutorial for `Classifying Names with
a Character-Level RNN <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>`_
in fairseq. It is recommended to quickly skim that tutorial before beginning
this one.

This tutorial covers:

1. **Preprocessing the data** to create dictionaries.
2. **Registering a new Model** that encodes an input sentence with a simple RNN
   and predicts the output label.
3. **Registering a new Task** that loads our dictionaries and dataset.
4. **Training the Model** using the existing command-line tools.
5. **Writing an evaluation script** that imports fairseq and allows us to
   interactively evaluate our model on new inputs.


1. Preprocessing the data
-------------------------

The original tutorial provides raw data, but we'll work with a modified version
of the data that is already tokenized into characters and split into separate
train, valid and test sets.

Download and extract the data from here:
`tutorial_names.tar.gz <https://dl.fbaipublicfiles.com/fairseq/data/tutorial_names.tar.gz>`_

Once extracted, let's preprocess the data using the :ref:`fairseq-preprocess`
command-line tool to create the dictionaries. While this tool is primarily
intended for sequence-to-sequence problems, we're able to reuse it here by
treating the label as a "target" sequence of length 1. We'll also output the
preprocessed files in "raw" format using the ``--dataset-impl`` option to
enhance readability:

.. code-block:: console

  > fairseq-preprocess \
    --trainpref names/train --validpref names/valid --testpref names/test \
    --source-lang input --target-lang label \
    --destdir names-bin --dataset-impl raw

After running the above command you should see a new directory,
:file:`names-bin/`, containing the dictionaries for *inputs* and *labels*.


2. Registering a new Model
--------------------------

Next we'll register a new model in fairseq that will encode an input sentence
with a simple RNN and predict the output label. Compared to the original PyTorch
tutorial, our version will also work with batches of data and GPU Tensors.

First let's copy the simple RNN module implemented in the `PyTorch tutorial
<https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#creating-the-network>`_.
Create a new file named :file:`fairseq/models/rnn_classifier.py` with the
following contents::

    import torch
    import torch.nn as nn

    class RNN(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()

            self.hidden_size = hidden_size

            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(input_size + hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)

We must also *register* this model with fairseq using the
:func:`~fairseq.models.register_model` function decorator. Once the model is
registered we'll be able to use it with the existing :ref:`Command-line Tools`.

All registered models must implement the :class:`~fairseq.models.BaseFairseqModel`
interface, so we'll create a small wrapper class in the same file and register
it in fairseq with the name ``'rnn_classifier'``::

    from fairseq.models import BaseFairseqModel, register_model

    # Note: the register_model "decorator" should immediately precede the
    # definition of the Model class.

    @register_model('rnn_classifier')
    class FairseqRNNClassifier(BaseFairseqModel):

        @staticmethod
        def add_args(parser):
            # Models can override this method to add new command-line arguments.
            # Here we'll add a new command-line argument to configure the
            # dimensionality of the hidden state.
            parser.add_argument(
                '--hidden-dim', type=int, metavar='N',
                help='dimensionality of the hidden state',
            )

        @classmethod
        def build_model(cls, args, task):
            # Fairseq initializes models by calling the ``build_model()``
            # function. This provides more flexibility, since the returned model
            # instance can be of a different type than the one that was called.
            # In this case we'll just return a FairseqRNNClassifier instance.

            # Initialize our RNN module
            rnn = RNN(
                # We'll define the Task in the next section, but for now just
                # notice that the task holds the dictionaries for the "source"
                # (i.e., the input sentence) and "target" (i.e., the label).
                input_size=len(task.source_dictionary),
                hidden_size=args.hidden_dim,
                output_size=len(task.target_dictionary),
            )

            # Return the wrapped version of the module
            return FairseqRNNClassifier(
                rnn=rnn,
                input_vocab=task.source_dictionary,
            )

        def __init__(self, rnn, input_vocab):
            super(FairseqRNNClassifier, self).__init__()

            self.rnn = rnn
            self.input_vocab = input_vocab

            # The RNN module in the tutorial expects one-hot inputs, so we can
            # precompute the identity matrix to help convert from indices to
            # one-hot vectors. We register it as a buffer so that it is moved to
            # the GPU when ``cuda()`` is called.
            self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))

        def forward(self, src_tokens, src_lengths):
            # The inputs to the ``forward()`` function are determined by the
            # Task, and in particular the ``'net_input'`` key in each
            # mini-batch. We'll define the Task in the next section, but for
            # now just know that *src_tokens* has shape `(batch, src_len)` and
            # *src_lengths* has shape `(batch)`.
            bsz, max_src_len = src_tokens.size()

            # Initialize the RNN hidden state. Compared to the original PyTorch
            # tutorial we'll also handle batched inputs and work on the GPU.
            hidden = self.rnn.initHidden()
            hidden = hidden.repeat(bsz, 1)  # expand for batched inputs
            hidden = hidden.to(src_tokens.device)  # move to GPU

            for i in range(max_src_len):
                # WARNING: The inputs have padding, so we should mask those
                # elements here so that padding doesn't affect the results.
                # This is left as an exercise for the reader. The padding symbol
                # is given by ``self.input_vocab.pad()`` and the unpadded length
                # of each input is given by *src_lengths*.

                # One-hot encode a batch of input characters.
                input = self.one_hot_inputs[src_tokens[:, i].long()]

                # Feed the input to our RNN.
                output, hidden = self.rnn(input, hidden)

            # Return the final output state for making a prediction
            return output

Finally let's define a *named architecture* with the configuration for our
model. This is done with the :func:`~fairseq.models.register_model_architecture`
function decorator. Thereafter this named architecture can be used with the
``--arch`` command-line argument, e.g., ``--arch pytorch_tutorial_rnn``::

    from fairseq.models import register_model_architecture

    # The first argument to ``register_model_architecture()`` should be the name
    # of the model we registered above (i.e., 'rnn_classifier'). The function we
    # register here should take a single argument *args* and modify it in-place
    # to match the desired architecture.

    @register_model_architecture('rnn_classifier', 'pytorch_tutorial_rnn')
    def pytorch_tutorial_rnn(args):
        # We use ``getattr()`` to prioritize arguments that are explicitly given
        # on the command-line, so that the defaults defined below are only used
        # when no other value has been specified.
        args.hidden_dim = getattr(args, 'hidden_dim', 128)


3. Registering a new Task
-------------------------

Now we'll register a new :class:`~fairseq.tasks.FairseqTask` that will load our
dictionaries and dataset. Tasks can also control how the data is batched into
mini-batches, but in this tutorial we'll reuse the batching provided by
:class:`fairseq.data.LanguagePairDataset`.

Create a new file named :file:`fairseq/tasks/simple_classification.py` with the
following contents::

  import os
  import torch

  from fairseq.data import Dictionary, LanguagePairDataset
  from fairseq.tasks import FairseqTask, register_task


  @register_task('simple_classification')
  class SimpleClassificationTask(FairseqTask):

      @staticmethod
      def add_args(parser):
          # Add some command-line arguments for specifying where the data is
          # located and the maximum supported input length.
          parser.add_argument('data', metavar='FILE',
                              help='file prefix for data')
          parser.add_argument('--max-positions', default=1024, type=int,
                              help='max input length')

      @classmethod
      def setup_task(cls, args, **kwargs):
          # Here we can perform any setup required for the task. This may include
          # loading Dictionaries, initializing shared Embedding layers, etc.
          # In this case we'll just load the Dictionaries.
          input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
          label_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
          print('| [input] dictionary: {} types'.format(len(input_vocab)))
          print('| [label] dictionary: {} types'.format(len(label_vocab)))

          return SimpleClassificationTask(args, input_vocab, label_vocab)

      def __init__(self, args, input_vocab, label_vocab):
          super().__init__(args)
          self.input_vocab = input_vocab
          self.label_vocab = label_vocab

      def load_dataset(self, split, **kwargs):
          """Load a given dataset split (e.g., train, valid, test)."""

          prefix = os.path.join(self.args.data, '{}.input-label'.format(split))

          # Read input sentences.
          sentences, lengths = [], []
          with open(prefix + '.input', encoding='utf-8') as file:
              for line in file:
                  sentence = line.strip()

                  # Tokenize the sentence, splitting on spaces
                  tokens = self.input_vocab.encode_line(
                      sentence, add_if_not_exist=False,
                  )

                  sentences.append(tokens)
                  lengths.append(tokens.numel())

          # Read labels.
          labels = []
          with open(prefix + '.label', encoding='utf-8') as file:
              for line in file:
                  label = line.strip()
                  labels.append(
                      # Convert label to a numeric ID.
                      torch.LongTensor([self.label_vocab.add_symbol(label)])
                  )

          assert len(sentences) == len(labels)
          print('| {} {} {} examples'.format(self.args.data, split, len(sentences)))

          # We reuse LanguagePairDataset since classification can be modeled as a
          # sequence-to-sequence task where the target sequence has length 1.
          self.datasets[split] = LanguagePairDataset(
              src=sentences,
              src_sizes=lengths,
              src_dict=self.input_vocab,
              tgt=labels,
              tgt_sizes=torch.ones(len(labels)),  # targets have length 1
              tgt_dict=self.label_vocab,
              left_pad_source=False,
              # Since our target is a single class label, there's no need for
              # teacher forcing. If we set this to ``True`` then our Model's
              # ``forward()`` method would receive an additional argument called
              # *prev_output_tokens* that would contain a shifted version of the
              # target sequence.
              input_feeding=False,
          )

      def max_positions(self):
          """Return the max input length allowed by the task."""
          # The source should be less than *args.max_positions* and the "target"
          # has max length 1.
          return (self.args.max_positions, 1)

      @property
      def source_dictionary(self):
          """Return the source :class:`~fairseq.data.Dictionary`."""
          return self.input_vocab

      @property
      def target_dictionary(self):
          """Return the target :class:`~fairseq.data.Dictionary`."""
          return self.label_vocab

      # We could override this method if we wanted more control over how batches
      # are constructed, but it's not necessary for this tutorial since we can
      # reuse the batching provided by LanguagePairDataset.
      #
      # def get_batch_iterator(
      #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
      #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
      #     seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1,
      #     data_buffer_size=0, disable_iterator_cache=False,
      # ):
      #     (...)


4. Training the Model
---------------------

Now we're ready to train the model. We can use the existing :ref:`fairseq-train`
command-line tool for this, making sure to specify our new Task (``--task
simple_classification``) and Model architecture (``--arch
pytorch_tutorial_rnn``):

.. note::

  You can also configure the dimensionality of the hidden state by passing the
  ``--hidden-dim`` argument to :ref:`fairseq-train`.

.. code-block:: console

  > fairseq-train names-bin \
    --task simple_classification \
    --arch pytorch_tutorial_rnn \
    --optimizer adam --lr 0.001 --lr-shrink 0.5 \
    --max-tokens 1000
  (...)
  | epoch 027 | loss 1.200 | ppl 2.30 | wps 15728 | ups 119.4 | wpb 116 | bsz 116 | num_updates 3726 | lr 1.5625e-05 | gnorm 1.290 | clip 0% | oom 0 | wall 32 | train_wall 21
  | epoch 027 | valid on 'valid' subset | valid_loss 1.41304 | valid_ppl 2.66 | num_updates 3726 | best 1.41208
  | done training in 31.6 seconds

The model files should appear in the :file:`checkpoints/` directory.


5. Writing an evaluation script
-------------------------------

Finally we can write a short script to evaluate our model on new inputs. Create
a new file named :file:`eval_classifier.py` with the following contents::

  from fairseq import checkpoint_utils, data, options, tasks

  # Parse command-line arguments for generation
  parser = options.get_generation_parser(default_task='simple_classification')
  args = options.parse_args_and_arch(parser)

  # Setup task
  task = tasks.setup_task(args)

  # Load model
  print('| loading model from {}'.format(args.path))
  models, _model_args = checkpoint_utils.load_model_ensemble([args.path], task=task)
  model = models[0]

  while True:
      sentence = input('\nInput: ')

      # Tokenize into characters
      chars = ' '.join(list(sentence.strip()))
      tokens = task.source_dictionary.encode_line(
          chars, add_if_not_exist=False,
      )

      # Build mini-batch to feed to the model
      batch = data.language_pair_dataset.collate(
          samples=[{'id': -1, 'source': tokens}],  # bsz = 1
          pad_idx=task.source_dictionary.pad(),
          eos_idx=task.source_dictionary.eos(),
          left_pad_source=False,
          input_feeding=False,
      )

      # Feed batch to the model and get predictions
      preds = model(**batch['net_input'])

      # Print top 3 predictions and their log-probabilities
      top_scores, top_labels = preds[0].topk(k=3)
      for score, label_idx in zip(top_scores, top_labels):
          label_name = task.target_dictionary.string([label_idx])
          print('({:.2f})\t{}'.format(score, label_name))

Now we can evaluate our model interactively. Note that we have included the
original data path (:file:`names-bin/`) so that the dictionaries can be loaded:

.. code-block:: console

  > python eval_classifier.py names-bin --path checkpoints/checkpoint_best.pt
  | [input] dictionary: 64 types
  | [label] dictionary: 24 types
  | loading model from checkpoints/checkpoint_best.pt

  Input: Satoshi
  (-0.61) Japanese
  (-1.20) Arabic
  (-2.86) Italian

  Input: Sinbad
  (-0.30) Arabic
  (-1.76) English
  (-4.08) Russian
