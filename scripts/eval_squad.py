#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Evaluate the perplexity of a trained language model.
"""

import json
import collections
import math
from os import path
import subprocess
import tempfile
import torch
import sys
sys.path.append('/private/home/yinhanliu/fairseq-py-huggingface')
sys.path.append('/private/home/yinhanliu/pytorch-pretrained-BERT')
from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize, BasicTokenizer
from fairseq import options, progress_bar, tasks, utils

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def get_final_text(pred_text, orig_text, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=True)
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        #if verbose_logging:
        #print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        #print("Length not equal after stripping spaces: '%s' vs '%s'",
        #              orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        #if verbose_logging:
        #print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def eval_dataset(task, model, dataset, data_file, args, use_cuda=True):
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 4096,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(model.max_positions()),
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        ignore_invalid_inputs=False,
    ).next_epoch_itr(shuffle=False)

    was_training = model.training
    model.eval()

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "token_to_orig_map"])

    example_index_to_features = collections.defaultdict(list)
    total = 0
    with torch.no_grad(), progress_bar.build_progress_bar(args, itr) as t:
        for batch in t:
            if use_cuda:
                batch = utils.move_to_cuda(batch)
            start_res, end_res, _  = model(**batch['net_input'])

            for start_logits, end_logits, id, text, mask, orig, idx_map, token_is_max_context in zip(start_res, end_res, batch['squad_ids'],
                                                              batch['net_input']['text'],
                                                              batch['net_input']['paragraph_mask'],
                                                              batch['actual_txt'],
                                                              batch['idx_map'],
                                                              batch['token_is_max_context']):
                example_index_to_features[id].append((start_logits, end_logits, text, mask, orig, idx_map, token_is_max_context))
                total += 1
        for id in example_index_to_features:
            prelim_predictions = []
            for span_idx, unique_example in enumerate(example_index_to_features[id]):
                start_logits, end_logits, text, mask, orig, idx_map, token_is_max_context = unique_example
                start_indexes = _get_best_indexes(start_logits, args.n_best_size)
                end_indexes = _get_best_indexes(end_logits, args.n_best_size)
                token_to_orig_map = {}
                mnz = mask.nonzero()
                start_idx = mnz[0]
                end_idx = mnz[-1]
                for j in range(start_idx, end_idx+1):
                    token_to_orig_map[j] = idx_map[j-start_idx]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(text) or start_index < start_idx or start_index > end_idx:
                            continue
                        if end_index >= len(text) or end_index < start_idx or end_index > end_idx:
                            continue
                        if end_index < start_index:
                            continue
                        if  not token_is_max_context[start_index]:
                            continue
                        length = end_index - start_index + 1
                        if length > 30:
                            continue

                        prelim_predictions.append(
                               _PrelimPrediction(
                                  feature_index=id,
                                  start_index=start_index,
                                  end_index=end_index,
                                  start_logit=start_logits[start_index],
                                  end_logit=end_logits[end_index],
                                  token_to_orig_map=token_to_orig_map))
            prelim_predictions = sorted(prelim_predictions,key=lambda x: (x.start_logit + x.end_logit),reverse=True)
            _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
            doc_tokens = orig.split()
            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= args.n_best_size:
                    break
                if pred.start_index > 0:
                    tok_tokens = [task.dictionary[ii]  for ii in text[pred.start_index:(pred.end_index + 1)]]
                    tok_text = " ".join(tok_tokens)
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")
                    tok_text = " ".join(tok_text.strip().split())
                    orig_doc_start = pred.token_to_orig_map[pred.start_index]
                    orig_doc_end = pred.token_to_orig_map[pred.end_index]
                    orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    final_text = get_final_text(tok_text, " ".join(orig_tokens))
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True
                nbest.append(_NbestPrediction(text=final_text,start_logit=pred.start_logit,end_logit=pred.end_logit))
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
            assert len(nbest) >= 1
            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1
                
            all_predictions[id] = nbest_json[0]["text"]

    with tempfile.NamedTemporaryFile('w') as f:
        json.dump(all_predictions, f)
        f.flush()
        try:
            res = subprocess.check_output(
               ["python", "official_squad_eval.py", data_file, f.name], #, '--na-prob-file', na_f.name],
               cwd=path.dirname(path.realpath(__file__)))
        except subprocess.CalledProcessError as e:
            res = e.output
        print(res.decode('utf-8'))

    if was_training:
        model.train()


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(parsed_args.path.split(':'), task)

    assert len(models) == 1

    model = models[0]
    if use_cuda:
        model.cuda()
    for arg in vars(parsed_args).keys():
        if arg not in {'concat_sentences_mode'}:
            setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data,  args.gen_subset, len(task.dataset(args.gen_subset))))

    data_file = path.join(args.data, args.data_file)
    eval_dataset(task, model, task.dataset(parsed_args.gen_subset), data_file, args, use_cuda)


if __name__ == '__main__':
    parser = options.get_parser('Evaluate SQUAD', 'squad')
    options.add_common_eval_args(parser)
    options.add_dataset_args(parser, gen=True)
    parser.add_argument('--data-file', type=str, help='the json data file to score (assumed to be in data dir)')
    parser.add_argument('--n-best-size', type=int, default=20)
    args = options.parse_args_and_arch(parser)
    main(args)
