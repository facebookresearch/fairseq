.. _Command-line Tools:

Command-line Tools
==================

Fairseq provides several command-line tools for training and evaluating models:

- :ref:`preprocess.py`: Data pre-processing: build vocabularies and binarize training data
- :ref:`train.py`: Train a new model on one or multiple GPUs
- :ref:`generate.py`: Translate pre-processed data with a trained model
- :ref:`interactive.py`: Translate raw text with a trained model
- :ref:`score.py`: BLEU scoring of generated translations against reference translations
- :ref:`eval_lm.py`: Language model evaluation


.. _preprocess.py:

preprocess.py
~~~~~~~~~~~~~
.. automodule:: preprocess

    .. argparse::
        :module: preprocess
        :func: get_parser
        :prog: preprocess.py


.. _train.py:

train.py
~~~~~~~~
.. automodule:: train

    .. argparse::
        :module: fairseq.options
        :func: get_training_parser
        :prog: train.py


.. _generate.py:

generate.py
~~~~~~~~~~~
.. automodule:: generate

    .. argparse::
        :module: fairseq.options
        :func: get_generation_parser
        :prog: generate.py


.. _interactive.py:

interactive.py
~~~~~~~~~~~~~~
.. automodule:: interactive

    .. argparse::
        :module: fairseq.options
        :func: get_interactive_generation_parser
        :prog: interactive.py


.. _score.py:

score.py
~~~~~~~~
.. automodule:: score

    .. argparse::
        :module: score
        :func: get_parser
        :prog: score.py


.. _eval_lm.py:

eval_lm.py
~~~~~~~~~~
.. automodule:: eval_lm

    .. argparse::
        :module: fairseq.options
        :func: get_eval_lm_parser
        :prog: eval_lm.py
