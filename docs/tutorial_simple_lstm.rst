Tutorial: Simple LSTM
=====================

In this tutorial we will extend fairseq by adding a new
:class:`~fairseq.models.FairseqEncoderDecoderModel` that encodes a source
sentence with an LSTM and then passes the final hidden state to a second LSTM
that decodes the target sentence (without attention).

This tutorial covers:

1. **Writing an Encoder and Decoder** to encode/decode the source/target
   sentence, respectively.
2. **Registering a new Model** so that it can be used with the existing
   :ref:`Command-line tools`.
3. **Training the Model** using the existing command-line tools.
4. **Making generation faster** by modifying the Decoder to use
   :ref:`Incremental decoding`.


1. Building an Encoder and Decoder
----------------------------------

In this section we'll define a simple LSTM Encoder and Decoder. All Encoders
should implement the :class:`~fairseq.models.FairseqEncoder` interface and
Decoders should implement the :class:`~fairseq.models.FairseqDecoder` interface.
These interfaces themselves extend :class:`torch.nn.Module`, so FairseqEncoders
and FairseqDecoders can be written and used in the same ways as ordinary PyTorch
Modules.


Encoder
~~~~~~~

Our Encoder will embed the tokens in the source sentence, feed them to a
:class:`torch.nn.LSTM` and return the final hidden state. To create our encoder
save the following in a new file named :file:`fairseq/models/simple_lstm.py`::

  import torch.nn as nn
  from fairseq import utils
  from fairseq.models import FairseqEncoder

  class SimpleLSTMEncoder(FairseqEncoder):

      def __init__(
          self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1,
      ):
          super().__init__(dictionary)
          self.args = args

          # Our encoder will embed the inputs before feeding them to the LSTM.
          self.embed_tokens = nn.Embedding(
              num_embeddings=len(dictionary),
              embedding_dim=embed_dim,
              padding_idx=dictionary.pad(),
          )
          self.dropout = nn.Dropout(p=dropout)

          # We'll use a single-layer, unidirectional LSTM for simplicity.
          self.lstm = nn.LSTM(
              input_size=embed_dim,
              hidden_size=hidden_dim,
              num_layers=1,
              bidirectional=False,
          )

      def forward(self, src_tokens, src_lengths):
          # The inputs to the ``forward()`` function are determined by the
          # Task, and in particular the ``'net_input'`` key in each
          # mini-batch. We discuss Tasks in the next tutorial, but for now just
          # know that *src_tokens* has shape `(batch, src_len)` and *src_lengths*
          # has shape `(batch)`.

          # Note that the source is typically padded on the left. This can be
          # configured by adding the `--left-pad-source "False"` command-line
          # argument, but here we'll make the Encoder handle either kind of
          # padding by converting everything to be right-padded.
          if self.args.left_pad_source:
              # Convert left-padding to right-padding.
              src_tokens = utils.convert_padding_direction(
                  src_tokens,
                  padding_idx=self.dictionary.pad(),
                  left_to_right=True
              )

          # Embed the source.
          x = self.embed_tokens(src_tokens)

          # Apply dropout.
          x = self.dropout(x)

          # Pack the sequence into a PackedSequence object to feed to the LSTM.
          x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

          # Get the output from the LSTM.
          _outputs, (final_hidden, _final_cell) = self.lstm(x)

          # Return the Encoder's output. This can be any object and will be
          # passed directly to the Decoder.
          return {
              # this will have shape `(bsz, hidden_dim)`
              'final_hidden': final_hidden.squeeze(0),
          }

      # Encoders are required to implement this method so that we can rearrange
      # the order of the batch elements during inference (e.g., beam search).
      def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          final_hidden = encoder_out['final_hidden']
          return {
              'final_hidden': final_hidden.index_select(0, new_order),
          }


Decoder
~~~~~~~

Our Decoder will predict the next word, conditioned on the Encoder's final
hidden state and an embedded representation of the previous target word -- which
is sometimes called *teacher forcing*. More specifically, we'll use a
:class:`torch.nn.LSTM` to produce a sequence of hidden states that we'll project
to the size of the output vocabulary to predict each target word.

::

  import torch
  from fairseq.models import FairseqDecoder

  class SimpleLSTMDecoder(FairseqDecoder):

      def __init__(
          self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128,
          dropout=0.1,
      ):
          super().__init__(dictionary)

          # Our decoder will embed the inputs before feeding them to the LSTM.
          self.embed_tokens = nn.Embedding(
              num_embeddings=len(dictionary),
              embedding_dim=embed_dim,
              padding_idx=dictionary.pad(),
          )
          self.dropout = nn.Dropout(p=dropout)

          # We'll use a single-layer, unidirectional LSTM for simplicity.
          self.lstm = nn.LSTM(
              # For the first layer we'll concatenate the Encoder's final hidden
              # state with the embedded target tokens.
              input_size=encoder_hidden_dim + embed_dim,
              hidden_size=hidden_dim,
              num_layers=1,
              bidirectional=False,
          )

          # Define the output projection.
          self.output_projection = nn.Linear(hidden_dim, len(dictionary))

      # During training Decoders are expected to take the entire target sequence
      # (shifted right by one position) and produce logits over the vocabulary.
      # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
      # ``dictionary.eos()``, followed by the target sequence.
      def forward(self, prev_output_tokens, encoder_out):
          """
          Args:
              prev_output_tokens (LongTensor): previous decoder outputs of shape
                  `(batch, tgt_len)`, for teacher forcing
              encoder_out (Tensor, optional): output from the encoder, used for
                  encoder-side attention

          Returns:
              tuple:
                  - the last decoder layer's output of shape
                    `(batch, tgt_len, vocab)`
                  - the last decoder layer's attention weights of shape
                    `(batch, tgt_len, src_len)`
          """
          bsz, tgt_len = prev_output_tokens.size()

          # Extract the final hidden state from the Encoder.
          final_encoder_hidden = encoder_out['final_hidden']

          # Embed the target sequence, which has been shifted right by one
          # position and now starts with the end-of-sentence symbol.
          x = self.embed_tokens(prev_output_tokens)

          # Apply dropout.
          x = self.dropout(x)

          # Concatenate the Encoder's final hidden state to *every* embedded
          # target token.
          x = torch.cat(
              [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
              dim=2,
          )

          # Using PackedSequence objects in the Decoder is harder than in the
          # Encoder, since the targets are not sorted in descending length order,
          # which is a requirement of ``pack_padded_sequence()``. Instead we'll
          # feed nn.LSTM directly.
          initial_state = (
              final_encoder_hidden.unsqueeze(0),  # hidden
              torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
          )
          output, _ = self.lstm(
              x.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
              initial_state,
          )
          x = output.transpose(0, 1)  # convert to shape `(bsz, tgt_len, hidden)`

          # Project the outputs to the size of the vocabulary.
          x = self.output_projection(x)

          # Return the logits and ``None`` for the attention weights
          return x, None


2. Registering the Model
------------------------

Now that we've defined our Encoder and Decoder we must *register* our model with
fairseq using the :func:`~fairseq.models.register_model` function decorator.
Once the model is registered we'll be able to use it with the existing
:ref:`Command-line Tools`.

All registered models must implement the
:class:`~fairseq.models.BaseFairseqModel` interface. For sequence-to-sequence
models (i.e., any model with a single Encoder and Decoder), we can instead
implement the :class:`~fairseq.models.FairseqEncoderDecoderModel` interface.

Create a small wrapper class in the same file and register it in fairseq with
the name ``'simple_lstm'``::

  from fairseq.models import FairseqEncoderDecoderModel, register_model

  # Note: the register_model "decorator" should immediately precede the
  # definition of the Model class.

  @register_model('simple_lstm')
  class SimpleLSTMModel(FairseqEncoderDecoderModel):

      @staticmethod
      def add_args(parser):
          # Models can override this method to add new command-line arguments.
          # Here we'll add some new command-line arguments to configure dropout
          # and the dimensionality of the embeddings and hidden states.
          parser.add_argument(
              '--encoder-embed-dim', type=int, metavar='N',
              help='dimensionality of the encoder embeddings',
          )
          parser.add_argument(
              '--encoder-hidden-dim', type=int, metavar='N',
              help='dimensionality of the encoder hidden state',
          )
          parser.add_argument(
              '--encoder-dropout', type=float, default=0.1,
              help='encoder dropout probability',
          )
          parser.add_argument(
              '--decoder-embed-dim', type=int, metavar='N',
              help='dimensionality of the decoder embeddings',
          )
          parser.add_argument(
              '--decoder-hidden-dim', type=int, metavar='N',
              help='dimensionality of the decoder hidden state',
          )
          parser.add_argument(
              '--decoder-dropout', type=float, default=0.1,
              help='decoder dropout probability',
          )

      @classmethod
      def build_model(cls, args, task):
          # Fairseq initializes models by calling the ``build_model()``
          # function. This provides more flexibility, since the returned model
          # instance can be of a different type than the one that was called.
          # In this case we'll just return a SimpleLSTMModel instance.

          # Initialize our Encoder and Decoder.
          encoder = SimpleLSTMEncoder(
              args=args,
              dictionary=task.source_dictionary,
              embed_dim=args.encoder_embed_dim,
              hidden_dim=args.encoder_hidden_dim,
              dropout=args.encoder_dropout,
          )
          decoder = SimpleLSTMDecoder(
              dictionary=task.target_dictionary,
              encoder_hidden_dim=args.encoder_hidden_dim,
              embed_dim=args.decoder_embed_dim,
              hidden_dim=args.decoder_hidden_dim,
              dropout=args.decoder_dropout,
          )
          model = SimpleLSTMModel(encoder, decoder)

          # Print the model architecture.
          print(model)

          return model

      # We could override the ``forward()`` if we wanted more control over how
      # the encoder and decoder interact, but it's not necessary for this
      # tutorial since we can inherit the default implementation provided by
      # the FairseqEncoderDecoderModel base class, which looks like:
      #
      # def forward(self, src_tokens, src_lengths, prev_output_tokens):
      #     encoder_out = self.encoder(src_tokens, src_lengths)
      #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
      #     return decoder_out

Finally let's define a *named architecture* with the configuration for our
model. This is done with the :func:`~fairseq.models.register_model_architecture`
function decorator. Thereafter this named architecture can be used with the
``--arch`` command-line argument, e.g., ``--arch tutorial_simple_lstm``::

  from fairseq.models import register_model_architecture

  # The first argument to ``register_model_architecture()`` should be the name
  # of the model we registered above (i.e., 'simple_lstm'). The function we
  # register here should take a single argument *args* and modify it in-place
  # to match the desired architecture.

  @register_model_architecture('simple_lstm', 'tutorial_simple_lstm')
  def tutorial_simple_lstm(args):
      # We use ``getattr()`` to prioritize arguments that are explicitly given
      # on the command-line, so that the defaults defined below are only used
      # when no other value has been specified.
      args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
      args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
      args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
      args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)


3. Training the Model
---------------------

Now we're ready to train the model. We can use the existing :ref:`fairseq-train`
command-line tool for this, making sure to specify our new Model architecture
(``--arch tutorial_simple_lstm``).

.. note::

  Make sure you've already preprocessed the data from the IWSLT example in the
  :file:`examples/translation/` directory.

.. code-block:: console

  > fairseq-train data-bin/iwslt14.tokenized.de-en \
    --arch tutorial_simple_lstm \
    --encoder-dropout 0.2 --decoder-dropout 0.2 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 12000
  (...)
  | epoch 052 | loss 4.027 | ppl 16.30 | wps 420805 | ups 39.7 | wpb 9841 | bsz 400 | num_updates 20852 | lr 1.95313e-05 | gnorm 0.218 | clip 0% | oom 0 | wall 529 | train_wall 396
  | epoch 052 | valid on 'valid' subset | valid_loss 4.74989 | valid_ppl 26.91 | num_updates 20852 | best 4.74954

The model files should appear in the :file:`checkpoints/` directory. While this
model architecture is not very good, we can use the :ref:`fairseq-generate` script to
generate translations and compute our BLEU score over the test set:

.. code-block:: console

  > fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
  (...)
  | Translated 6750 sentences (153132 tokens) in 17.3s (389.12 sentences/s, 8827.68 tokens/s)
  | Generate test with beam=5: BLEU4 = 8.18, 38.8/12.1/4.7/2.0 (BP=1.000, ratio=1.066, syslen=139865, reflen=131146)


4. Making generation faster
---------------------------

While autoregressive generation from sequence-to-sequence models is inherently
slow, our implementation above is especially slow because it recomputes the
entire sequence of Decoder hidden states for every output token (i.e., it is
``O(n^2)``). We can make this significantly faster by instead caching the
previous hidden states.

In fairseq this is called :ref:`Incremental decoding`. Incremental decoding is a
special mode at inference time where the Model only receives a single timestep
of input corresponding to the immediately previous output token (for teacher
forcing) and must produce the next output incrementally. Thus the model must
cache any long-term state that is needed about the sequence, e.g., hidden
states, convolutional states, etc.

To implement incremental decoding we will modify our model to implement the
:class:`~fairseq.models.FairseqIncrementalDecoder` interface. Compared to the
standard :class:`~fairseq.models.FairseqDecoder` interface, the incremental
decoder interface allows ``forward()`` methods to take an extra keyword argument
(*incremental_state*) that can be used to cache state across time-steps.

Let's replace our ``SimpleLSTMDecoder`` with an incremental one::

  import torch
  from fairseq.models import FairseqIncrementalDecoder

  class SimpleLSTMDecoder(FairseqIncrementalDecoder):

      def __init__(
          self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128,
          dropout=0.1,
      ):
          # This remains the same as before.
          super().__init__(dictionary)
          self.embed_tokens = nn.Embedding(
              num_embeddings=len(dictionary),
              embedding_dim=embed_dim,
              padding_idx=dictionary.pad(),
          )
          self.dropout = nn.Dropout(p=dropout)
          self.lstm = nn.LSTM(
              input_size=encoder_hidden_dim + embed_dim,
              hidden_size=hidden_dim,
              num_layers=1,
              bidirectional=False,
          )
          self.output_projection = nn.Linear(hidden_dim, len(dictionary))

      # We now take an additional kwarg (*incremental_state*) for caching the
      # previous hidden and cell states.
      def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
          if incremental_state is not None:
              # If the *incremental_state* argument is not ``None`` then we are
              # in incremental inference mode. While *prev_output_tokens* will
              # still contain the entire decoded prefix, we will only use the
              # last step and assume that the rest of the state is cached.
              prev_output_tokens = prev_output_tokens[:, -1:]

          # This remains the same as before.
          bsz, tgt_len = prev_output_tokens.size()
          final_encoder_hidden = encoder_out['final_hidden']
          x = self.embed_tokens(prev_output_tokens)
          x = self.dropout(x)
          x = torch.cat(
              [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
              dim=2,
          )

          # We will now check the cache and load the cached previous hidden and
          # cell states, if they exist, otherwise we will initialize them to
          # zeros (as before). We will use the ``utils.get_incremental_state()``
          # and ``utils.set_incremental_state()`` helpers.
          initial_state = utils.get_incremental_state(
              self, incremental_state, 'prev_state',
          )
          if initial_state is None:
              # first time initialization, same as the original version
              initial_state = (
                  final_encoder_hidden.unsqueeze(0),  # hidden
                  torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
              )

          # Run one step of our LSTM.
          output, latest_state = self.lstm(x.transpose(0, 1), initial_state)

          # Update the cache with the latest hidden and cell states.
          utils.set_incremental_state(
              self, incremental_state, 'prev_state', latest_state,
          )

          # This remains the same as before
          x = output.transpose(0, 1)
          x = self.output_projection(x)
          return x, None

      # The ``FairseqIncrementalDecoder`` interface also requires implementing a
      # ``reorder_incremental_state()`` method, which is used during beam search
      # to select and reorder the incremental state.
      def reorder_incremental_state(self, incremental_state, new_order):
          # Load the cached state.
          prev_state = utils.get_incremental_state(
              self, incremental_state, 'prev_state',
          )

          # Reorder batches according to *new_order*.
          reordered_state = (
              prev_state[0].index_select(1, new_order),  # hidden
              prev_state[1].index_select(1, new_order),  # cell
          )

          # Update the cached state.
          utils.set_incremental_state(
              self, incremental_state, 'prev_state', reordered_state,
          )

Finally, we can rerun generation and observe the speedup:

.. code-block:: console

  # Before

  > fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
  (...)
  | Translated 6750 sentences (153132 tokens) in 17.3s (389.12 sentences/s, 8827.68 tokens/s)
  | Generate test with beam=5: BLEU4 = 8.18, 38.8/12.1/4.7/2.0 (BP=1.000, ratio=1.066, syslen=139865, reflen=131146)

  # After

  > fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
  (...)
  | Translated 6750 sentences (153132 tokens) in 5.5s (1225.54 sentences/s, 27802.94 tokens/s)
  | Generate test with beam=5: BLEU4 = 8.18, 38.8/12.1/4.7/2.0 (BP=1.000, ratio=1.066, syslen=139865, reflen=131146)
