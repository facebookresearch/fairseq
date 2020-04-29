# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . agent import Agent
from . import DEFAULT_EOS, GET, SEND
from fairseq import checkpoint_utils, utils, tasks
import os
import json


class SimulTransAgent(Agent):
    def __init__(self, args):
        # Load Model
        self.load_model(args)

        # build word spliter
        self.build_word_splitter(args)

        self.max_len = args.max_len

        self.eos = DEFAULT_EOS

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--user-dir", type=str, default="example/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--src-splitter-type", type=str, default=None,
                            help="Subword splitter type for source text")
        parser.add_argument("--tgt-splitter-type", type=str, default=None,
                            help="Subword splitter type for target text")
        parser.add_argument("--src-splitter-path", type=str, default=None,
                            help="Subword splitter model path for source text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--max-len", type=int, default=150,
                            help="Maximum length difference between source and target prediction")
        parser.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                            help='A dictionary used to override model args at generation '
                                 'that were used during model training')
        # fmt: on
        return parser

    def load_dictionary(self, task):
        raise NotImplementedError

    def load_model(self, args):
        args.user_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        utils.import_user_module(args)
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename, json.loads(args.model_overrides))

        saved_args = state["args"]
        saved_args.data = args.data_bin

        task = tasks.setup_task(saved_args)

        # build model for ensemble
        self.model = task.build_model(saved_args)
        self.model.load_state_dict(state["model"], strict=True)

        # Set dictionary
        self.load_dictionary(task)

    def init_states(self):
        return {
            "indices": {"src": [], "tgt": []},
            "tokens": {"src": [], "tgt": []},
            "segments": {"src": [], "tgt": []},
            "steps": {"src": 0, "tgt": 0},
            "finished": False,
            "finish_read": False,
            "model_states": {}
        }

    def update_states(self, states, new_state):
        raise NotImplementedError

    def policy(self, states):
        # Read and Write policy
        action = None

        while action is None:
            if states["finished"]:
                # Finish the hypo by sending eos to server
                return self.finish_action()

            # Model make decision given current states
            decision = self.model.decision_from_states(states)

            if decision == 0 and not self.finish_read(states):
                # READ
                action = self.read_action(states)
            else:
                # WRITE
                action = self.write_action(states)

            # None means we make decision again but not sending server anything
            # This happened when read a bufffered token
            # Or predict a subword
        return action

    def finish_read(self, states):
        raise NotImplementedError

    def write_action(self, states):
        token, index = self.model.predict_from_states(states)

        if index == self.dict["tgt"].eos() or len(states["tokens"]["tgt"]) > self.max_len:
            # Finish this sentence is predict EOS
            states["finished"] = True
            end_idx_last_full_word = self._target_length(states)

        else:
            states["tokens"]["tgt"] += [token]
            end_idx_last_full_word = (
                self.word_splitter["tgt"]
                .end_idx_last_full_word(states["tokens"]["tgt"])
            )
            self._append_indices(states, [index], "tgt")

        if end_idx_last_full_word > states["steps"]["tgt"]:
            # Only sent detokenized full words to the server
            word = self.word_splitter["tgt"].merge(
                states["tokens"]["tgt"][
                    states["steps"]["tgt"]: end_idx_last_full_word
                ]
            )
            states["steps"]["tgt"] = end_idx_last_full_word
            states["segments"]["tgt"] += [word]

            return {'key': SEND, 'value': word}
        else:
            return None

    def read_action(self, states):
        return {'key': GET, 'value': None}

    def finish_action(self):
        return {'key': SEND, 'value': DEFAULT_EOS}

    def reset(self):
        pass

    def finish_eval(self, states, new_state):
        if len(new_state) == 0 and len(states["indices"]["src"]) == 0:
            return True
        return False

    def _append_indices(self, states, new_indices, key):
        states["indices"][key] += new_indices

    def _target_length(self, states):
        return len(states["tokens"]['tgt'])
