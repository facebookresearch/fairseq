# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import subprocess
from contextlib import redirect_stdout

from fairseq import options
from fairseq_cli import eval_lm, preprocess


def reprocess(fle):
    # takes in a file of generate.py translation generate_output
    # returns a source dict and hypothesis dict, where keys are the ID num (as a string)
    # and values and the corresponding source and translation. There may be several translations
    # per source, so the values for hypothesis_dict are lists.
    # parses output of generate.py

    with open(fle, "r") as f:
        txt = f.read()

    """reprocess generate.py output"""
    p = re.compile(r"[STHP][-]\d+\s*")
    hp = re.compile(r"(\s*[-]?\d+[.]?\d+\s*)|(\s*(-inf)\s*)")
    source_dict = {}
    hypothesis_dict = {}
    score_dict = {}
    target_dict = {}
    pos_score_dict = {}
    lines = txt.split("\n")

    for line in lines:
        line += "\n"
        prefix = re.search(p, line)
        if prefix is not None:
            assert len(prefix.group()) > 2, "prefix id not found"
            _, j = prefix.span()
            id_num = prefix.group()[2:]
            id_num = int(id_num)
            line_type = prefix.group()[0]
            if line_type == "H":
                h_txt = line[j:]
                hypo = re.search(hp, h_txt)
                assert (
                    hypo is not None
                ), "regular expression failed to find the hypothesis scoring"
                _, i = hypo.span()
                score = hypo.group()
                if id_num in hypothesis_dict:
                    hypothesis_dict[id_num].append(h_txt[i:])
                    score_dict[id_num].append(float(score))
                else:
                    hypothesis_dict[id_num] = [h_txt[i:]]
                    score_dict[id_num] = [float(score)]

            elif line_type == "S":
                source_dict[id_num] = line[j:]
            elif line_type == "T":
                target_dict[id_num] = line[j:]
            elif line_type == "P":
                pos_scores = (line[j:]).split()
                pos_scores = [float(x) for x in pos_scores]
                if id_num in pos_score_dict:
                    pos_score_dict[id_num].append(pos_scores)
                else:
                    pos_score_dict[id_num] = [pos_scores]

    return source_dict, hypothesis_dict, score_dict, target_dict, pos_score_dict


def reprocess_nbest(fle):
    """reprocess interactive.py output"""
    with open(fle, "r") as f:
        txt = f.read()

    source_dict = {}
    hypothesis_dict = {}
    score_dict = {}
    target_dict = {}
    pos_score_dict = {}
    lines = txt.split("\n")

    hp = re.compile(r"[-]?\d+[.]?\d+")
    j = -1

    for _i, line in enumerate(lines):
        line += "\n"
        line_type = line[0]

        if line_type == "H":
            hypo = re.search(hp, line)
            _, start_index = hypo.span()
            score = hypo.group()
            if j in score_dict:
                score_dict[j].append(float(score))
                hypothesis_dict[j].append(line[start_index:].strip("\t"))
            else:
                score_dict[j] = [float(score)]
                hypothesis_dict[j] = [line[start_index:].strip("\t")]
        elif line_type == "O":
            j += 1
            source_dict[j] = line[2:]
            # we don't have the targets for interactive.py
            target_dict[j] = "filler"

        elif line_type == "P":
            pos_scores = [float(pos_score) for pos_score in line.split()[1:]]
            if j in pos_score_dict:
                pos_score_dict[j].append(pos_scores)
            else:
                pos_score_dict[j] = [pos_scores]

    assert source_dict.keys() == hypothesis_dict.keys()
    assert source_dict.keys() == pos_score_dict.keys()
    assert source_dict.keys() == score_dict.keys()

    return source_dict, hypothesis_dict, score_dict, target_dict, pos_score_dict


def write_reprocessed(
    sources,
    hypos,
    targets,
    source_outfile,
    hypo_outfile,
    target_outfile,
    right_to_left=False,
    prefix_len=None,
    bpe_symbol=None,
    target_prefix_frac=None,
    source_prefix_frac=None,
):

    """writes nbest hypothesis for rescoring"""
    assert not (
        prefix_len is not None and target_prefix_frac is not None
    ), "in writing reprocessed, only one type of prefix may be used"
    assert not (
        prefix_len is not None and source_prefix_frac is not None
    ), "in writing reprocessed, only one type of prefix may be used"
    assert not (
        target_prefix_frac is not None and source_prefix_frac is not None
    ), "in writing reprocessed, only one type of prefix may be used"

    with open(source_outfile, "w") as source_file, open(
        hypo_outfile, "w"
    ) as hypo_file, open(target_outfile, "w") as target_file:

        assert len(sources) == len(hypos), "sources and hypos list length mismatch"
        if right_to_left:
            for i in range(len(sources)):
                for j in range(len(hypos[i])):
                    if prefix_len is None:
                        hypo_file.write(make_right_to_left(hypos[i][j]) + "\n")
                    else:
                        raise NotImplementedError()
                    source_file.write(make_right_to_left(sources[i]) + "\n")
                    target_file.write(make_right_to_left(targets[i]) + "\n")
        else:
            for i in sorted(sources.keys()):
                for j in range(len(hypos[i])):
                    if prefix_len is not None:
                        shortened = (
                            get_prefix_no_bpe(hypos[i][j], bpe_symbol, prefix_len)
                            + "\n"
                        )
                        hypo_file.write(shortened)
                        source_file.write(sources[i])
                        target_file.write(targets[i])
                    elif target_prefix_frac is not None:
                        num_words, shortened, num_bpe_tokens = calc_length_from_frac(
                            hypos[i][j], target_prefix_frac, bpe_symbol
                        )
                        shortened += "\n"
                        hypo_file.write(shortened)
                        source_file.write(sources[i])
                        target_file.write(targets[i])
                    elif source_prefix_frac is not None:
                        num_words, shortened, num_bpe_tokensn = calc_length_from_frac(
                            sources[i], source_prefix_frac, bpe_symbol
                        )
                        shortened += "\n"
                        hypo_file.write(hypos[i][j])
                        source_file.write(shortened)
                        target_file.write(targets[i])
                    else:
                        hypo_file.write(hypos[i][j])
                        source_file.write(sources[i])
                        target_file.write(targets[i])


def calc_length_from_frac(bpe_sentence, prefix_frac, bpe_symbol):
    # return number of words, (not bpe tokens) that we want
    no_bpe_sen = remove_bpe(bpe_sentence, bpe_symbol)
    len_sen = len(no_bpe_sen.split())

    num_words = math.ceil(len_sen * prefix_frac)
    prefix = get_prefix_no_bpe(bpe_sentence, bpe_symbol, num_words)
    num_bpe_tokens = len(prefix.split())
    return num_words, prefix, num_bpe_tokens


def get_prefix(sentence, prefix_len):
    """assuming no bpe, gets the prefix of the sentence with prefix_len words"""
    tokens = sentence.strip("\n").split()
    if prefix_len >= len(tokens):
        return sentence.strip("\n")
    else:
        return " ".join(tokens[:prefix_len])


def get_prefix_no_bpe(sentence, bpe_symbol, prefix_len):
    if bpe_symbol is None:
        return get_prefix(sentence, prefix_len)
    else:
        return " ".join(get_prefix_from_len(sentence.split(), bpe_symbol, prefix_len))


def get_prefix_from_len(sentence, bpe_symbol, prefix_len):
    """get the prefix of sentence with bpe, with prefix len in terms of words, not bpe tokens"""
    bpe_count = sum([bpe_symbol.strip(" ") in t for t in sentence[:prefix_len]])
    if bpe_count == 0:
        return sentence[:prefix_len]
    else:
        return sentence[:prefix_len] + get_prefix_from_len(
            sentence[prefix_len:], bpe_symbol, bpe_count
        )


def get_num_bpe_tokens_from_len(sentence, bpe_symbol, prefix_len):
    """given a prefix length in terms of words, return the number of bpe tokens"""
    prefix = get_prefix_no_bpe(sentence, bpe_symbol, prefix_len)
    assert len(remove_bpe(prefix, bpe_symbol).split()) <= prefix_len
    return len(prefix.split(" "))


def make_right_to_left(line):
    tokens = line.split()
    tokens.reverse()
    new_line = " ".join(tokens)
    return new_line


def remove_bpe(line, bpe_symbol):
    line = line.replace("\n", "")
    line = (line + " ").replace(bpe_symbol, "").rstrip()
    return line + ("\n")


def remove_bpe_dict(pred_dict, bpe_symbol):
    new_dict = {}
    for i in pred_dict:
        if type(pred_dict[i]) == list:
            new_list = [remove_bpe(elem, bpe_symbol) for elem in pred_dict[i]]
            new_dict[i] = new_list
        else:
            new_dict[i] = remove_bpe(pred_dict[i], bpe_symbol)
    return new_dict


def parse_bleu_scoring(line):
    p = re.compile(r"(BLEU4 = )\d+[.]\d+")
    res = re.search(p, line)
    assert res is not None, line
    return float(res.group()[8:])


def get_full_from_prefix(hypo_prefix, hypos):
    """given a hypo prefix, recover the first hypo from the list of complete hypos beginning with that prefix"""
    for hypo in hypos:
        hypo_prefix = hypo_prefix.strip("\n")
        len_prefix = len(hypo_prefix)
        if hypo[:len_prefix] == hypo_prefix:
            return hypo
    # no match found
    raise Exception()


def get_score(
    a,
    b,
    c,
    target_len,
    bitext_score1,
    bitext_score2=None,
    lm_score=None,
    lenpen=None,
    src_len=None,
    tgt_len=None,
    bitext1_backwards=False,
    bitext2_backwards=False,
    normalize=False,
):
    if bitext1_backwards:
        bitext1_norm = src_len
    else:
        bitext1_norm = tgt_len
    if bitext_score2 is not None:
        if bitext2_backwards:
            bitext2_norm = src_len
        else:
            bitext2_norm = tgt_len
    else:
        bitext2_norm = 1
        bitext_score2 = 0
    if normalize:
        score = (
            a * bitext_score1 / bitext1_norm
            + b * bitext_score2 / bitext2_norm
            + c * lm_score / src_len
        )
    else:
        score = a * bitext_score1 + b * bitext_score2 + c * lm_score

    if lenpen is not None:
        score /= (target_len) ** float(lenpen)

    return score


class BitextOutput(object):
    def __init__(
        self,
        output_file,
        backwards,
        right_to_left,
        bpe_symbol,
        prefix_len=None,
        target_prefix_frac=None,
        source_prefix_frac=None,
    ):
        """process output from rescoring"""
        source, hypo, score, target, pos_score = reprocess(output_file)
        if backwards:
            self.hypo_fracs = source_prefix_frac
        else:
            self.hypo_fracs = target_prefix_frac

        # remove length penalty so we can use raw scores
        score, num_bpe_tokens = get_score_from_pos(
            pos_score, prefix_len, hypo, bpe_symbol, self.hypo_fracs, backwards
        )
        source_lengths = {}
        target_lengths = {}

        assert hypo.keys() == source.keys(), "key mismatch"
        if backwards:
            tmp = hypo
            hypo = source
            source = tmp
        for i in source:
            # since we are reranking, there should only be one hypo per source sentence
            if backwards:
                len_src = len(source[i][0].split())
                # record length without <eos>
                if len_src == num_bpe_tokens[i][0] - 1:
                    source_lengths[i] = num_bpe_tokens[i][0] - 1
                else:
                    source_lengths[i] = num_bpe_tokens[i][0]

                target_lengths[i] = len(hypo[i].split())

                source[i] = remove_bpe(source[i][0], bpe_symbol)
                target[i] = remove_bpe(target[i], bpe_symbol)
                hypo[i] = remove_bpe(hypo[i], bpe_symbol)

                score[i] = float(score[i][0])
                pos_score[i] = pos_score[i][0]

            else:
                len_tgt = len(hypo[i][0].split())
                # record length without <eos>
                if len_tgt == num_bpe_tokens[i][0] - 1:
                    target_lengths[i] = num_bpe_tokens[i][0] - 1
                else:
                    target_lengths[i] = num_bpe_tokens[i][0]

                source_lengths[i] = len(source[i].split())

                if right_to_left:
                    source[i] = remove_bpe(make_right_to_left(source[i]), bpe_symbol)
                    target[i] = remove_bpe(make_right_to_left(target[i]), bpe_symbol)
                    hypo[i] = remove_bpe(make_right_to_left(hypo[i][0]), bpe_symbol)
                    score[i] = float(score[i][0])
                    pos_score[i] = pos_score[i][0]
                else:
                    assert (
                        len(hypo[i]) == 1
                    ), "expected only one hypothesis per source sentence"
                    source[i] = remove_bpe(source[i], bpe_symbol)
                    target[i] = remove_bpe(target[i], bpe_symbol)
                    hypo[i] = remove_bpe(hypo[i][0], bpe_symbol)
                    score[i] = float(score[i][0])
                    pos_score[i] = pos_score[i][0]

        self.rescore_source = source
        self.rescore_hypo = hypo
        self.rescore_score = score
        self.rescore_target = target
        self.rescore_pos_score = pos_score
        self.backwards = backwards
        self.right_to_left = right_to_left
        self.target_lengths = target_lengths
        self.source_lengths = source_lengths


class BitextOutputFromGen(object):
    def __init__(
        self,
        predictions_bpe_file,
        bpe_symbol=None,
        nbest=False,
        prefix_len=None,
        target_prefix_frac=None,
    ):
        if nbest:
            (
                pred_source,
                pred_hypo,
                pred_score,
                pred_target,
                pred_pos_score,
            ) = reprocess_nbest(predictions_bpe_file)
        else:
            pred_source, pred_hypo, pred_score, pred_target, pred_pos_score = reprocess(
                predictions_bpe_file
            )

        assert len(pred_source) == len(pred_hypo)
        assert len(pred_source) == len(pred_score)
        assert len(pred_source) == len(pred_target)
        assert len(pred_source) == len(pred_pos_score)

        # remove length penalty so we can use raw scores
        pred_score, num_bpe_tokens = get_score_from_pos(
            pred_pos_score, prefix_len, pred_hypo, bpe_symbol, target_prefix_frac, False
        )

        self.source = pred_source
        self.target = pred_target
        self.score = pred_score
        self.pos_score = pred_pos_score
        self.hypo = pred_hypo
        self.target_lengths = {}
        self.source_lengths = {}

        self.no_bpe_source = remove_bpe_dict(pred_source.copy(), bpe_symbol)
        self.no_bpe_hypo = remove_bpe_dict(pred_hypo.copy(), bpe_symbol)
        self.no_bpe_target = remove_bpe_dict(pred_target.copy(), bpe_symbol)

        # indexes to match those from the rescoring models
        self.rescore_source = {}
        self.rescore_target = {}
        self.rescore_pos_score = {}
        self.rescore_hypo = {}
        self.rescore_score = {}
        self.num_hypos = {}
        self.backwards = False
        self.right_to_left = False

        index = 0

        for i in sorted(pred_source.keys()):
            for j in range(len(pred_hypo[i])):

                self.target_lengths[index] = len(self.hypo[i][j].split())
                self.source_lengths[index] = len(self.source[i].split())

                self.rescore_source[index] = self.no_bpe_source[i]
                self.rescore_target[index] = self.no_bpe_target[i]
                self.rescore_hypo[index] = self.no_bpe_hypo[i][j]
                self.rescore_score[index] = float(pred_score[i][j])
                self.rescore_pos_score[index] = pred_pos_score[i][j]
                self.num_hypos[index] = len(pred_hypo[i])
                index += 1


def get_score_from_pos(
    pos_score_dict, prefix_len, hypo_dict, bpe_symbol, hypo_frac, backwards
):
    score_dict = {}
    num_bpe_tokens_dict = {}
    assert prefix_len is None or hypo_frac is None
    for key in pos_score_dict:
        score_dict[key] = []
        num_bpe_tokens_dict[key] = []
        for i in range(len(pos_score_dict[key])):
            if prefix_len is not None and not backwards:
                num_bpe_tokens = get_num_bpe_tokens_from_len(
                    hypo_dict[key][i], bpe_symbol, prefix_len
                )
                score_dict[key].append(sum(pos_score_dict[key][i][:num_bpe_tokens]))
                num_bpe_tokens_dict[key].append(num_bpe_tokens)
            elif hypo_frac is not None:
                num_words, shortened, hypo_prefix_len = calc_length_from_frac(
                    hypo_dict[key][i], hypo_frac, bpe_symbol
                )
                score_dict[key].append(sum(pos_score_dict[key][i][:hypo_prefix_len]))
                num_bpe_tokens_dict[key].append(hypo_prefix_len)
            else:
                score_dict[key].append(sum(pos_score_dict[key][i]))
                num_bpe_tokens_dict[key].append(len(pos_score_dict[key][i]))
    return score_dict, num_bpe_tokens_dict


class LMOutput(object):
    def __init__(
        self,
        lm_score_file,
        lm_dict=None,
        prefix_len=None,
        bpe_symbol=None,
        target_prefix_frac=None,
    ):
        (
            lm_sentences,
            lm_sen_scores,
            lm_sen_pos_scores,
            lm_no_bpe_sentences,
            lm_bpe_tokens,
        ) = parse_lm(
            lm_score_file,
            prefix_len=prefix_len,
            bpe_symbol=bpe_symbol,
            target_prefix_frac=target_prefix_frac,
        )

        self.sentences = lm_sentences
        self.score = lm_sen_scores
        self.pos_score = lm_sen_pos_scores
        self.lm_dict = lm_dict
        self.no_bpe_sentences = lm_no_bpe_sentences
        self.bpe_tokens = lm_bpe_tokens


def parse_lm(input_file, prefix_len=None, bpe_symbol=None, target_prefix_frac=None):
    """parse output of eval_lm"""
    with open(input_file, "r") as f:
        text = f.readlines()
        text = text[7:]
        cleaned_text = text[:-2]

        sentences = {}
        sen_scores = {}
        sen_pos_scores = {}
        no_bpe_sentences = {}
        num_bpe_tokens_dict = {}
        for _i, line in enumerate(cleaned_text):
            tokens = line.split()
            if tokens[0].isdigit():
                line_id = int(tokens[0])
                scores = [float(x[1:-1]) for x in tokens[2::2]]
                sentences[line_id] = " ".join(tokens[1::2][:-1]) + "\n"
                if bpe_symbol is not None:
                    # exclude <eos> symbol to match output from generate.py
                    bpe_sen = " ".join(tokens[1::2][:-1]) + "\n"
                    no_bpe_sen = remove_bpe(bpe_sen, bpe_symbol)
                    no_bpe_sentences[line_id] = no_bpe_sen

                if prefix_len is not None:
                    num_bpe_tokens = get_num_bpe_tokens_from_len(
                        bpe_sen, bpe_symbol, prefix_len
                    )
                    sen_scores[line_id] = sum(scores[:num_bpe_tokens])
                    num_bpe_tokens_dict[line_id] = num_bpe_tokens
                elif target_prefix_frac is not None:
                    num_words, shortened, target_prefix_len = calc_length_from_frac(
                        bpe_sen, target_prefix_frac, bpe_symbol
                    )
                    sen_scores[line_id] = sum(scores[:target_prefix_len])
                    num_bpe_tokens_dict[line_id] = target_prefix_len
                else:
                    sen_scores[line_id] = sum(scores)
                    num_bpe_tokens_dict[line_id] = len(scores)

                sen_pos_scores[line_id] = scores

    return sentences, sen_scores, sen_pos_scores, no_bpe_sentences, num_bpe_tokens_dict


def get_directories(
    data_dir_name,
    num_rescore,
    gen_subset,
    fw_name,
    shard_id,
    num_shards,
    sampling=False,
    prefix_len=None,
    target_prefix_frac=None,
    source_prefix_frac=None,
):
    nbest_file_id = (
        "nbest_"
        + str(num_rescore)
        + "_subset_"
        + gen_subset
        + "_fw_name_"
        + fw_name
        + "_shard_"
        + str(shard_id)
        + "_of_"
        + str(num_shards)
    )

    if sampling:
        nbest_file_id += "_sampling"

    # the directory containing all information for this nbest list
    pre_gen = (
        os.path.join(os.path.dirname(__file__))
        + "/rerank_data/"
        + data_dir_name
        + "/"
        + nbest_file_id
    )
    # the directory to store the preprocessed nbest list, for left to right rescoring
    left_to_right_preprocessed_dir = pre_gen + "/left_to_right_preprocessed"
    if source_prefix_frac is not None:
        left_to_right_preprocessed_dir = (
            left_to_right_preprocessed_dir + "/prefix_frac" + str(source_prefix_frac)
        )
    # the directory to store the preprocessed nbest list, for right to left rescoring
    right_to_left_preprocessed_dir = pre_gen + "/right_to_left_preprocessed"
    # the directory to store the preprocessed nbest list, for backwards rescoring
    backwards_preprocessed_dir = pre_gen + "/backwards"
    if target_prefix_frac is not None:
        backwards_preprocessed_dir = (
            backwards_preprocessed_dir + "/prefix_frac" + str(target_prefix_frac)
        )
    elif prefix_len is not None:
        backwards_preprocessed_dir = (
            backwards_preprocessed_dir + "/prefix_" + str(prefix_len)
        )

    # the directory to store the preprocessed nbest list, for rescoring with P(T)
    lm_preprocessed_dir = pre_gen + "/lm_preprocessed"

    return (
        pre_gen,
        left_to_right_preprocessed_dir,
        right_to_left_preprocessed_dir,
        backwards_preprocessed_dir,
        lm_preprocessed_dir,
    )


def lm_scoring(
    preprocess_directory,
    bpe_status,
    gen_output,
    pre_gen,
    cur_lm_dict,
    cur_lm_name,
    cur_language_model,
    cur_lm_bpe_code,
    batch_size,
    lm_score_file,
    target_lang,
    source_lang,
    prefix_len=None,
):
    if prefix_len is not None:
        assert (
            bpe_status == "different"
        ), "bpe status must be different to use prefix len"
    if bpe_status == "no bpe":
        # run lm on output without bpe
        write_reprocessed(
            gen_output.no_bpe_source,
            gen_output.no_bpe_hypo,
            gen_output.no_bpe_target,
            pre_gen + "/rescore_data_no_bpe.de",
            pre_gen + "/rescore_data_no_bpe.en",
            pre_gen + "/reference_file_no_bpe",
        )

        preprocess_lm_param = [
            "--only-source",
            "--trainpref",
            pre_gen + "/rescore_data_no_bpe." + target_lang,
            "--srcdict",
            cur_lm_dict,
            "--destdir",
            preprocess_directory,
        ]
        preprocess_parser = options.get_preprocessing_parser()
        input_args = preprocess_parser.parse_args(preprocess_lm_param)
        preprocess.main(input_args)

        eval_lm_param = [
            preprocess_directory,
            "--path",
            cur_language_model,
            "--output-word-probs",
            "--batch-size",
            str(batch_size),
            "--max-tokens",
            "1024",
            "--sample-break-mode",
            "eos",
            "--gen-subset",
            "train",
        ]

        eval_lm_parser = options.get_eval_lm_parser()
        input_args = options.parse_args_and_arch(eval_lm_parser, eval_lm_param)

        with open(lm_score_file, "w") as f:
            with redirect_stdout(f):
                eval_lm.main(input_args)

    elif bpe_status == "shared":
        preprocess_lm_param = [
            "--only-source",
            "--trainpref",
            pre_gen + "/rescore_data." + target_lang,
            "--srcdict",
            cur_lm_dict,
            "--destdir",
            preprocess_directory,
        ]
        preprocess_parser = options.get_preprocessing_parser()
        input_args = preprocess_parser.parse_args(preprocess_lm_param)
        preprocess.main(input_args)

        eval_lm_param = [
            preprocess_directory,
            "--path",
            cur_language_model,
            "--output-word-probs",
            "--batch-size",
            str(batch_size),
            "--sample-break-mode",
            "eos",
            "--gen-subset",
            "train",
        ]

        eval_lm_parser = options.get_eval_lm_parser()
        input_args = options.parse_args_and_arch(eval_lm_parser, eval_lm_param)

        with open(lm_score_file, "w") as f:
            with redirect_stdout(f):
                eval_lm.main(input_args)

    elif bpe_status == "different":
        rescore_file = pre_gen + "/rescore_data_no_bpe"
        rescore_bpe = pre_gen + "/rescore_data_new_bpe"

        rescore_file += "."
        rescore_bpe += "."

        write_reprocessed(
            gen_output.no_bpe_source,
            gen_output.no_bpe_hypo,
            gen_output.no_bpe_target,
            rescore_file + source_lang,
            rescore_file + target_lang,
            pre_gen + "/reference_file_no_bpe",
            bpe_symbol=None,
        )

        # apply LM bpe to nbest list
        bpe_src_param = [
            "-c",
            cur_lm_bpe_code,
            "--input",
            rescore_file + target_lang,
            "--output",
            rescore_bpe + target_lang,
        ]
        subprocess.call(
            [
                "python",
                os.path.join(
                    os.path.dirname(__file__), "subword-nmt/subword_nmt/apply_bpe.py"
                ),
            ]
            + bpe_src_param,
            shell=False,
        )
        # uncomment to use fastbpe instead of subword-nmt bpe
        # bpe_src_param = [rescore_bpe+target_lang, rescore_file+target_lang, cur_lm_bpe_code]
        # subprocess.call(["/private/home/edunov/fastBPE/fast", "applybpe"] + bpe_src_param, shell=False)

        preprocess_dir = preprocess_directory

        preprocess_lm_param = [
            "--only-source",
            "--trainpref",
            rescore_bpe + target_lang,
            "--srcdict",
            cur_lm_dict,
            "--destdir",
            preprocess_dir,
        ]
        preprocess_parser = options.get_preprocessing_parser()
        input_args = preprocess_parser.parse_args(preprocess_lm_param)
        preprocess.main(input_args)

        eval_lm_param = [
            preprocess_dir,
            "--path",
            cur_language_model,
            "--output-word-probs",
            "--batch-size",
            str(batch_size),
            "--max-tokens",
            "1024",
            "--sample-break-mode",
            "eos",
            "--gen-subset",
            "train",
        ]

        eval_lm_parser = options.get_eval_lm_parser()
        input_args = options.parse_args_and_arch(eval_lm_parser, eval_lm_param)

        with open(lm_score_file, "w") as f:
            with redirect_stdout(f):
                eval_lm.main(input_args)


def rescore_file_name(
    nbest_dir,
    prefix_len,
    scorer_name,
    lm_file=False,
    target_prefix_frac=None,
    source_prefix_frac=None,
    backwards=None,
):
    if lm_file:
        score_file = nbest_dir + "/lm_score_translations_model_" + scorer_name + ".txt"
    else:
        score_file = nbest_dir + "/" + scorer_name + "_score_translations.txt"
    if backwards:
        if prefix_len is not None:
            score_file += "prefix_len" + str(prefix_len)
        elif target_prefix_frac is not None:
            score_file += "target_prefix_frac" + str(target_prefix_frac)
    else:
        if source_prefix_frac is not None:
            score_file += "source_prefix_frac" + str(source_prefix_frac)
    return score_file
