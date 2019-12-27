#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import os

import sentencepiece as spm
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module
import json
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EOD_TOKEN = "EOD"
EOS_TOKEN = ""
DEFAULT_EOS_TOKEN = "</s>"

def add_online_argument(parser):
    parser.add_argument("--server", required=True)
    parser.add_argument("--src-spm", required=True)
    parser.add_argument("--tgt-spm", required=True)
    return parser


def check_args(args):
    assert args.path is not None, "--path required for generation!"
    assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=(1000000.0, 1000000.0),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id, src_len, factor
):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_words = sp.DecodePieces(hyp_pieces.split())
        print(
            "{}".format(hyp_pieces), file=res_files["hypo.units"]
        )
        print("{}".format(hyp_words), file=res_files["hypo.words"])

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = sp.DecodePieces(tgt_pieces.split())
        print("{}".format(tgt_pieces), file=res_files["ref.units"])
        print("{}".format(tgt_words), file=res_files["ref.words"])
        # only score top hypothesis

        hypo["fast_monotonic_step"] = hypo.get("fast_monotonic_step", [])
        delay = {
            "src_len" : src_len,
            "delays" : [ min([(1 + item) * factor, src_len]) for item in hypo["fast_monotonic_step"]] 
        }
        print(
            json.dumps(delay),
            file=res_files["hypo.delays"]
        )
        #if max(delay["delays"]) >= int(src_len):
        #    import pdb; pdb.set_trace()
        if not args.quiet:
            logger.debug("HYPO:" + hyp_words)
            logger.debug("TARGET:" + tgt_words)
            logger.debug("___________________")


def prepare_result_files(args):
    def get_res_file(file_prefix):
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
        "hypo.delays": get_res_file("hypo.delays"),
    }


def load_model_and_criterions(filenames, arg_overrides=None, task=None):
    model = []
    criterions = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=True)
        if getattr(task, "set_num_mel_bins", None) is not None:
            task.set_num_mel_bins(args.input_feat_per_channel)

        criterion = task.build_criterion(args)
        if "criterion" in state:
            criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return model, criterions, args


def optimize_models(args, use_cuda, model):
    """Optimize ensemble for generation
    """
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

class DummyClient():
    def __init__(self, args):
        self.input_file = args.server
        self.text = []
        self.results = defaultdict(list)
        with open(self.input_file) as f:
            for line in f:
                self.text.append(line.split())

        self._init_pointer()
    
    def _init_pointer(self):
        self.sent_id = 0
        self.tok_id = 0

    def read(self):
        if self.sent_id >= len(self.text):
            return EOD_TOKEN
        
        if self.tok_id >= len(self.text[self.sent_id]):
            return EOS_TOKEN

        token = self.text[self.sent_id][self.tok_id]
        self.tok_id += 1        
        return token
    
    def write(self, token, sent_id):
        if sent_id < self.sent_id:
            delay = len(self.text[sent_id])
        else:
            delay = self.tok_id

        self.results[sent_id].append(
            {
                "token" : token,
                "delay" : delay
            }
        )
    def finish_hypo(self):
        self.tok_id = 0
        self.sent_id += 1
    
    def finished(self):
        return self.sent_id >= len(self.text)

from client import SimulSTEvaluationService

class SimulTransEvalClient():
    def __init__(self, session=None):
        session = SimulSTEvaluationService('localhost', 12321) 
        self.set_session(session)
    
    def start(self):
        self.session.__enter__()

    def end(self):
        self.session.__exit__(None, None, None)

    def set_session(self, session):
        self.session = session

    def read(self):
        new_state = self.session.get_src()
        token = new_state.get("segment", DEFAULT_EOS_TOKEN)
        return token

    def write(self, token):
        self.session.send_hypo(token)
    
    def finished(self):
        return False

    def finish_hypo(self):
        self.session.send_hypo(DEFAULT_EOS_TOKEN)


def init_client(args):
    #return DummyClient(args)
    return SimulTransEvalClient()
    
def read_and_encode(client, spm_model, dictionary, buffer, use_cuda):
    raw_token = client.read()
    if raw_token not in [EOS_TOKEN, EOD_TOKEN, DEFAULT_EOS_TOKEN]:
        tokens = spm_model.EncodeAsPieces(raw_token)
        token_ids = dictionary.encode_line(
            tokens,
            line_tokenizer=lambda x : x,
            add_if_not_exist=False,
            append_eos=False
        ).unsqueeze(0).long()

        src_indices = buffer["src_indices"]
        if src_indices is not None:
            token_ids = token_ids.to(src_indices.device)
            src_indices = torch.cat(
                [src_indices ,token_ids]
                ,1
            )
        else:
            src_indices = token_ids
        
        if use_cuda:
            src_indices = src_indices.cuda()

        buffer["src_indices"] = src_indices
    buffer["src_txt"].append(raw_token)


def decode_and_write(client, spm_model, buffer):
    #print(pred_txt_buffer)
    #import pdb; pdb.set_trace()
    raw_token = spm_model.DecodePieces(buffer["tgt_subwords"])
    segment_id = len(buffer["tgt_txt"])
    buffer["tgt_txt"].append(raw_token) 
    client.write(raw_token)
    buffer["tgt_subwords"] = []


def is_begin_of_subword(token):
    return len(token) == 0 or token[0] == '\u2581' 

def main(args):
    check_args(args)
    import_user_module(args)

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    task = tasks.setup_task(args)

    logger.info("| decoding with criterion {}".format(args.criterion))

    # Load ensemble
    logger.info("| loading model(s) from {}".format(args.path))

    model, criterions, _model_args = load_model_and_criterions(
        args.path.split(":"),
        arg_overrides=eval(args.model_overrides),  # noqa
        task=task,
    )

    # Set dictionary
    tgt_dict = task.target_dictionary
    src_dict = task.source_dictionary

    optimize_models(args, use_cuda, model)

    generator = task.build_generator(args)

    max_length = args.max_len_b 

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    sp = {}
    for side in ["src", "tgt"]:
        sp[side] = spm.SentencePieceProcessor()
        sp[side].Load(getattr(args, f"{side}_spm"))

    def init_buffer():
        return {
            "src_indices" : None,
            "tgt_indices" : torch.LongTensor([[tgt_dict.eos()]]),
            "tgt_subwords" : [],
            "decoder_states" : [],
            "tgt_token" : None,
            "src_token" : None,
            "src_txt" : [],
            "tgt_txt" : []
        }

    client = init_client(args)
    client.start()

    while not client.finished():
        buffer = init_buffer()
        # At least read one token to start
        read_and_encode(client, sp["src"], src_dict, buffer, use_cuda)

        if buffer["src_txt"][-1] == DEFAULT_EOS_TOKEN:
            # There is nothing left, break loop
            break
        
        while buffer["tgt_token"] != EOS_TOKEN and len(buffer["tgt_txt"]) < max_length:
            # Choose Policy
            action = model.get_action(buffer)
            print(buffer["src_txt"])
            print(buffer["tgt_txt"], len(buffer["tgt_txt"]))

            if action == 0 and buffer["src_txt"][-1] != DEFAULT_EOS_TOKEN:
                # READ
                read_and_encode(client, sp["src"], src_dict, buffer, use_cuda)
            else:
                # WRITE
                lprobs = model.get_normalized_probs(
                    [buffer["decoder_states"][:, -1:]], 
                    log_probs=True
                )

                tgt_idx = lprobs.argmax(dim=-1)

                buffer["tgt_indices"] = torch.cat(
                    [buffer["tgt_indices"].to(tgt_idx.device), tgt_idx],
                    dim=1
                )

                # If finish translation a whole word
                tgt_token = tgt_dict.string(tgt_idx) 
                if is_begin_of_subword(tgt_token):
                    decode_and_write(client, sp["tgt"], buffer)
                
                buffer["tgt_subwords"].append(tgt_token)
                buffer["tgt_token"] = tgt_token

        client.finish_hypo()

    client.end()
            
def cli_main():
    parser = options.get_generation_parser()
    parser = add_online_argument(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
