#!/usr/bin/env python3

"""
Download data for trivia-question task.

Some parts of this script come from:
 * https://github.com/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb

The original codes is licensed under Apache 2.0.
"""

import datetime
import functools
import gzip
import json
import os
import pprint
import random
import string
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import t5
import tensorflow as tf
import tensorflow_datasets as tfds
import time

from contextlib import contextmanager
import logging as py_logging

DATA_DIR = "./data"

nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "nq-train.tsv"),
    "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
}


def nq_jsonl_to_tsv(in_fname, out_fname):
  def extract_answer(tokens, span):
    """Reconstruct answer from token span and remove extra spaces."""
    start, end = span["start_token"], span["end_token"]
    ans = " ".join(tokens[start:end])
    # Remove incorrect spacing around punctuation.
    ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
    ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
    ans = ans.replace("( ", "(").replace(" )", ")")
    ans = ans.replace("`` ", "\"").replace(" ''", "\"")
    ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
    return ans

  count = 0
  with tf.io.gfile.GFile(in_fname, "rb") as infile,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    for line in tqdm(gzip.open(infile)):
      ex = json.loads(line)
      # Remove any examples with more than one answer.
      if len(ex['annotations'][0]['short_answers']) != 1:
        continue
      # Questions in NQ do not include a question mark.
      question = ex["question_text"] + "?"
      answer_span = ex['annotations'][0]['short_answers'][0]
      # Handle the two document formats in NQ (tokens or text).
      if "document_tokens" in ex:
        tokens = [t["token"] for t in ex["document_tokens"]]
      elif "document_text" in ex:
        tokens = ex["document_text"].split(" ")
      answer = extract_answer(tokens, answer_span)
      # Write this line as <question>\t<answer>
      outfile.write("%s\t%s\n" % (question, answer))
      count += 1
    return count


def nq_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
  return ds


def trivia_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
      """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
      return {
          "inputs":
              tf.strings.join(
                  ["trivia question: ", normalize_text(ex["question"])]),
          "targets": normalize_text(ex["answer"])
      }
    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


def tiviaqa_extract_qa(ds):
  def exract_qa(ex):
    return {
        "question": ex["question"],
        "answer": ex["answer"]["value"]
    }
  return ds.map(exract_qa, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_nq_data():
    # Public directory of Natural Questions data on GCS.
    NQ_JSONL_DIR = "gs://natural_questions/v1.0-simplified/"
    NQ_SPLIT_FNAMES = {
        "train": "simplified-nq-train.jsonl.gz",
        "validation": "nq-dev-all.jsonl.gz"
    }
    nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
    nq_tsv_path = {
        "train": os.path.join(DATA_DIR, "nq-train.tsv"),
        "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
    }
    if tf.io.gfile.exists(nq_counts_path):
        # Used cached data and counts.
        num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
    else:
        # Create TSVs and get counts.
        num_nq_examples = {}
        for split, fname in NQ_SPLIT_FNAMES.items():
            num_nq_examples[split] = nq_jsonl_to_tsv(
                os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
        json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))

    t5.data.TaskRegistry.add(
        "nq_context_free",
        dataset_fn=nq_dataset_fn,
        splits=["train", "validation"],
        text_preprocessor=[trivia_preprocessor],
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        num_input_examples=num_nq_examples
)
    nq_task = t5.data.TaskRegistry.get("nq_context_free")

    for split in ('train', 'validation'):
        ds = nq_task.get_dataset(split=split, sequence_length={"inputs": 128, "targets": 32})
        with open(f'raw/{split}-nq.questions', 'wt') as question_file, open(f'raw/{split}-nq.answers', 'wt') as answer_file:
            for ex in ds.as_numpy_iterator():
                question_file.write(ex["inputs_plaintext"].decode("utf-8") + '\n')
                answer_file.write(ex["targets_plaintext"].decode("utf-8") + '\n')


def get_trivia_data():
    trivia_qa_ds = tfds.load(
        "trivia_qa/unfiltered.nocontext",
        data_dir=DATA_DIR.strip(),
        download_and_prepare_kwargs={"download_dir": "./downloads"},
    )

    t5.data.TaskRegistry.add(
        "triviaqa_context_free",
        # A TfdsTask takes in a TFDS name instead of a tf.data.Dataset function.
        t5.data.TfdsTask,
        tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
        tfds_data_dir=DATA_DIR,
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        text_preprocessor=[tiviaqa_extract_qa, trivia_preprocessor],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy]
    )

    triviaqa_task = t5.data.TaskRegistry.get("nq_context_free")

    for split in ('train', 'validation'):
        ds = triviaqa_task.get_dataset(split=split, sequence_length={"inputs": 128, "targets": 32})
        with open(f'raw/{split}-triviaqa.questions', 'wt') as question_file, open(f'raw/{split}-triviaqa.answers', 'wt') as answer_file:
            for ex in ds.as_numpy_iterator():
                question_file.write(ex["inputs_plaintext"].decode("utf-8") + '\n')
                answer_file.write(ex["targets_plaintext"].decode("utf-8") + '\n')


def main():
    """Main."""
    os.makedirs('raw', exist_ok=True)
    get_nq_data()
    get_trivia_data()


if __name__ == '__main__':
    main()
