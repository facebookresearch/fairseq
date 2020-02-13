from .agent import Agent
from . import DEFAULT_EOS, GET, SEND
from . import register_agent
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
from .word_splitter import *
import os


@register_agent("simul_trans_text")
class SimulTransTextAgent(Agent):
    def __init__(self, args):
        self.word_splitter = {}

        self.word_splitter["src"] = eval(f"{args.src_splitter_type}WordSplitter")(
                getattr(args, f"src_splitter_path")
            )
        self.word_splitter["tgt"] = eval(f"{args.tgt_splitter_type}WordSplitter")(
                getattr(args, f"tgt_splitter_path")
            )

        # Load Model
        self.load_model(args)

        self.max_len = args.max_len

    @staticmethod
    def add_args(parser):
        parser.add_argument('--model-path', type=str, default=None, 
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--user-dir", type=str,
                            help="User directory for simultaneous translation")
        parser.add_argument("--src-splitter-type", type=str,
                            help="")
        parser.add_argument("--tgt-splitter-type", type=str,
                            help="")
        parser.add_argument("--src-splitter-path", type=str,
                            help="")
        parser.add_argument("--tgt-splitter-path", type=str,
                            help="")
        parser.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                            help="source language")
        parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                            help="target language")
        parser.add_argument("--max-len", type=int, default=150,
                            help="Maximum length difference between source and target prediction")
        parser.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                        help='a dictionary used to override model args at generation '
                                'that were used during model training')
        return parser
    
    def load_model(self, args):
        args.user_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        utils.import_user_module(args)
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename, eval(args.model_overrides))

        args = state["args"]

        task = tasks.setup_task(args)

        # build model for ensemble
        self.model = task.build_model(args)
        self.model.load_state_dict(state["model"], strict=True)
        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def init_states(self):
        return {
            "indices": {"src": [], "tgt": []},
            "tokens" : {"src": [], "tgt": []},
            "words" : {"src": [], "tgt": []},
            "steps" : {"src": 0, "tgt": 0},
            "finished" : False,
        }

    def update_states(self, states, new_state):
        if len(new_state) == 0:
            return states

        new_word = new_state["segment"]

        # Split words and index the token
        if new_word not in [DEFAULT_EOS]:
            tokens = self.word_splitter["src"].split(new_word)
            # Get indices from dictionary
            # You can change to you own dictionary
            indices = self.dict["src"].encode_line(
                tokens,
                line_tokenizer=lambda x : x,
                add_if_not_exist=False,
                append_eos=False
            ).tolist()
        else:
            tokens = [new_word]
            indices = [self.dict["src"].eos()]

        # Update states
        states["words"]["src"] += [new_word]
        states["indices"]["src"] += indices
        states["tokens"]["src"] += tokens

        return states

    def policy(self, states):
        # Read and Write policy

        if states["finished"]:
            # Finish the hypo by sending eos to server
            return self.finish_action()
        
        action = None

        while action is None:
            # Model make decision given current states
            decision = self.model.decision_from_states(states)

            if decision == 0 and DEFAULT_EOS not in states["tokens"]["src"]:
                # READ
                action = self.read_action(states)
            else:
                # WRITE 
                action = self.write_action(states)

            # None means we make decision again but not sending server anything
            # This happened when read a bufffered token
            # Or predict a subword
        return action

    def write_action(self, states):
        token, index = self.model.predict_from_states(states)

        if index == self.dict["tgt"].eos() or len(states["tokens"]["tgt"]) > self.max_len:
            # Finish this sentence is predict EOS
            states["finished"] = True
            end_idx_last_full_word = len(states["tokens"]['tgt'])

        else:    
            states["tokens"]["tgt"] += [token]
            states["indices"]["tgt"] += [index]
            end_idx_last_full_word = (
                self.word_splitter["tgt"]
                .end_idx_last_full_word(states["tokens"]["tgt"])
            ) 

        if end_idx_last_full_word > states["steps"]["tgt"]:
            # Only sent detokenized full words to the server
            word = self.word_splitter["tgt"].merge(
                states["tokens"]["tgt"][
                    states["steps"]["tgt"]: end_idx_last_full_word
                ]
            )
            states["steps"]["tgt"] = end_idx_last_full_word
            states["words"]["tgt"] += [word] 

            return {'key': SEND, 'value': word}
        else:
            return None

    def read_action(self, states):
        # Ignore models decision on reading when EOS in src_tokens
        if DEFAULT_EOS in states["tokens"]["src"]:
            return None
        
        # Increase source step by one
        states["steps"]["src"] += 1

        # At leat one word is read
        if len(states["tokens"]["src"]) == 0:
            return {'key': GET, 'value': None} 

        # Only request new word if there is no buffered tokens
        if len(states["tokens"]["src"]) <= states["steps"]["src"]:
            return {'key': GET, 'value': None}

        return None

    def finish_action(self):
        return {'key': SEND, 'value': DEFAULT_EOS}

    def reset(self):
        pass

    def finish_eval(self, states, new_state):
        if len(new_state) == 0 and len(states["indices"]["src"]) == 0:
            return True
        return False
