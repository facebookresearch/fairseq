from . simul_trans_agent import SimulTransAgent
from . import DEFAULT_EOS, GET, SEND
from . import register_agent
from . word_splitter import *


@register_agent("simul_trans_text")
class SimulTransTextAgent(SimulTransAgent):
    def build_word_splitter(self, args):
        self.word_splitter = {}

        self.word_splitter["src"] = eval(f"{args.src_splitter_type}WordSplitter")(
                getattr(args, f"src_splitter_path")
            )
        self.word_splitter["tgt"] = eval(f"{args.tgt_splitter_type}WordSplitter")(
                getattr(args, f"tgt_splitter_path")
            )

    def load_dictionary(self, task):
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def update_states(self, states, new_state):

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
        states["segments"]["src"] += [new_word]
        states["tokens"]["src"] += tokens
        self._append_indices(states, indices, "src")

        return states

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