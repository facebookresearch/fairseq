# Transformer with Pointer-Generator Network

This page describes the `transformer_pointer_generator` model that incorporates
a pointing mechanism in the Transformer model that facilitates copying of input
words to the output. This architecture is described in [Enarvi et al. (2020)](https://www.aclweb.org/anthology/2020.nlpmc-1.4/).

## Background

The pointer-generator network was introduced in [See et al. (2017)](https://arxiv.org/abs/1704.04368)
for RNN encoder-decoder attention models. A similar mechanism can be
incorporated in a Transformer model by reusing one of the many attention
distributions for pointing. The attention distribution over the input words is
interpolated with the normal output distribution over the vocabulary words. This
allows the model to generate words that appear in the input, even if they don't
appear in the vocabulary, helping especially with small vocabularies.

## Implementation

The mechanism for copying out-of-vocabulary words from the input has been
implemented differently to See et al. In their [implementation](https://github.com/abisee/pointer-generator)
they convey the word identities through the model in order to be able to produce
words that appear in the input sequence but not in the vocabulary. A different
approach was taken in the Fairseq implementation to keep it self-contained in
the model file, avoiding any changes to the rest of the code base. Copying
out-of-vocabulary words is possible by pre-processing the input and
post-processing the output. This is described in detail in the next section.

## Usage

The training and evaluation procedure is outlined below. You can also find a
more detailed example for the XSum dataset on [this page](README.xsum.md).

##### 1. Create a vocabulary and extend it with source position markers

The pointing mechanism is especially helpful with small vocabularies, if we are
able to recover the identities of any out-of-vocabulary words that are copied
from the input. For this purpose, the model allows extending the vocabulary with
special tokens that can be used in place of `<unk>` tokens to identify different
input positions. For example, the user may add `<unk-0>`, `<unk-1>`, `<unk-2>`,
etc. to the end of the vocabulary, after the normal words. Below is an example
of how to create a vocabulary of 10000 most common words and add 1000 input
position markers.

```bash
vocab_size=10000
position_markers=1000
export LC_ALL=C
cat train.src train.tgt |
  tr -s '[:space:]' '\n' |
  sort |
  uniq -c |
  sort -k1,1bnr -k2 |
  head -n "$((vocab_size - 4))" |
  awk '{ print $2 " " $1 }' >dict.pg.txt
python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >>dict.pg.txt
```

##### 2. Preprocess the text data

The idea is that any `<unk>` tokens in the text are replaced with `<unk-0>` if
it appears in the first input position, `<unk-1>` if it appears in the second
input position, and so on. This can be achieved using the `preprocess.py` script
that is provided in this directory.

##### 3. Train a model

The number of these special tokens is given to the model with the
`--source-position-markers` argument—the model simply maps all of these to the
same word embedding as `<unk>`.

The attention distribution that is used for pointing is selected using the
`--alignment-heads` and `--alignment-layer` command-line arguments in the same
way as with the `transformer_align` model.

##### 4. Generate text and postprocess it

When using the model to generate text, you want to preprocess the input text in
the same way that training data was processed, replacing out-of-vocabulary words
with `<unk-N>` tokens. If any of these tokens are copied to the output, the
actual words can be retrieved from the unprocessed input text. Any `<unk-N>`
token should be replaced with the word at position N in the original input
sequence. This can be achieved using the `postprocess.py` script.
