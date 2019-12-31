DEFAULT_EOS = '</s>'
GET = 0
SEND = 1
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
import os
import sentencepiece as spm

class Agent(object):
    "an agent needs to follow this pattern"
    def __init__(self, *args, **kwargs):
        pass
    def init_states(self):
        return []

    def update_states(self, states, new_state):
        return states + [new_state]

    def finish_eval(self, states, new_state):
        if len(new_state) == 0 and len(states) == 0:
            return True
        return False
        
    def decode(self, session):
        states = self.init_states()
        self.reset()
        
        while True:
            
            # take an action
            action = self.policy(states)

            if action['key'] == GET:
                new_state = session.get_src()
                if self.finish_eval(states, new_state):
                    break # end of document
                states = self.update_states(states, new_state)

            elif action['key'] == SEND:
                session.send_hypo(action['value'])
                if action['value'] == DEFAULT_EOS:
                    states = self.init_states()  # clean the history, wait for next sentence
                    self.reset()
            else:
                raise NotImplementedError

    def policy(self, state: list) -> dict:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DummyWaitAgent(Agent):

    def __init__(self, k=2, *args, **kwargs):
        self.wait_k = k
        self.curr_k = 0

    def reset(self):
        self.curr_k = 0

    def policy(self, states):
        if len(states) - self.curr_k < self.wait_k:
            action = {'key': GET, 'value': None}
        else:
            action = {'key': SEND, 'value': states[self.curr_k]['segment']}
            self.curr_k += 1
        return action


class SimulTransAgent(Agent):
    def __init__(self, parser):
        args = options.parse_args_and_arch(parser)
        self.args = args
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args)


        # Load Model
        self.load_model(
            args.path,
            arg_overrides=eval(args.model_overrides),  # noqa
            task=task,
        )
        # Set dictionary
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary

        # Load SPM model
        self.src_spm = spm.SentencePieceProcessor()
        self.src_spm.Load(args.src_spm)
        self.tgt_spm = spm.SentencePieceProcessor()
        self.tgt_spm.Load(args.tgt_spm)

        self.max_len = args.max_len
    
    @staticmethod
    def argument_parser():
        parser = options.get_generation_parser()
        parser.add_argument("--src-spm", type=str,
                            help="Source side sentence piece model")
        parser.add_argument("--tgt-spm", type=str,
                            help="Target side sentence piece model")
        parser.add_argument("--max-len", type=int, default=150,
                            help="Maximum length difference between source and target prediction")
        return parser

    def load_model(self, filename, arg_overrides=None, task=None):
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        self.model = task.build_model(args)
        self.model.load_state_dict(state["model"], strict=True)

    def init_states(self):
        return {
            "src_indices" : None,
            "tgt_indices" : torch.LongTensor([[self.tgt_dict.eos()]]),
            "tgt_subwords" : [],
            "decoder_states" : [],
            "src_tokens" : [],
            "tgt_tokens" : [],
            "src_txt" : [],
            "tgt_txt" : [],
            "src_step" : 0,
            "tgt_step" : 0,
            "finished" : False,
        }

    def prepocess_state(self, state):
        if len(state) == 0:
            return state 
        
        raw_token = state["segment"]

        if raw_token not in [DEFAULT_EOS]:
            tokens = self.src_spm.EncodeAsPieces(raw_token)
            token_ids = self.src_dict.encode_line(
                tokens,
                line_tokenizer=lambda x : x,
                add_if_not_exist=False,
                append_eos=False
            ).unsqueeze(0).long()
        else:
            tokens = [raw_token]
            token_ids = torch.LongTensor([[self.src_dict.eos()]])

        if self.use_cuda:
            token_ids.cuda()
        
        return {
            "src_txt": raw_token, 
            "src_tokens": tokens, 
            "src_indices": token_ids, 
            "sent_id" : state["sent_id"],
            "segment_id" : state["segment_id"]
        }

    def update_states(self, states, new_state):
        if len(new_state) == 0:
            return states

        new_state_info = self.prepocess_state(new_state)

        if states["src_indices"] is not None:
            states["src_indices"] = torch.cat(
                [
                    states["src_indices"],
                    new_state_info["src_indices"]
                ]
                ,1
            )
        else:
            states["src_indices"] = new_state_info["src_indices"]
        
        states["src_txt"].append(new_state_info["src_txt"])
        states["src_tokens"] += new_state_info["src_tokens"]

        return states

    @staticmethod
    def is_begin_of_subword(token):
        return len(token) == 0 or token[0] == '\u2581' 
    
    @staticmethod
    def tgt_len_from_states(states):
        return states["tgt_indices"].size(1) - 1

    def max_length(self, states):
        return self.max_len

    def pred_one_target_token(self, states):
        src_tokens = states["src_indices"][:states["src_step"]]
        src_lengths = torch.LongTensor([src_tokens.numel()])

        tgt_tokens = states["tgt_indices"]
        tgt_tokens = tgt_tokens.to(src_tokens.device)

        self.model.eval()

        # Update encoder state
        encoder_outs = self.model.encoder(src_tokens, src_lengths) 

        # Generate decoder state
        decoder_states, _ = self.model.decoder(tgt_tokens, encoder_outs, states)

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], 
            log_probs=True
        )

        tgt_idx = lprobs.argmax(dim=-1)

        states["tgt_indices"] = torch.cat(
            [states["tgt_indices"].to(tgt_idx.device), tgt_idx],
            dim=1
        )

        tgt_token = self.tgt_dict.string(tgt_idx) 
        states["tgt_subwords"].append(tgt_token)

        return tgt_token, tgt_idx[0,0].item()

    def finish_eval(self, states, new_state):
        if len(new_state) == 0 and states["src_indices"] is None:
            return True
        return False

    def policy(self, states):
        # Read and Write policy
        if states["finished"]:
            # Finish the hypo by sending eos to server
            return {'key': SEND, 'value': DEFAULT_EOS}
        
        # Model make decision given current states
        decision = self.model.get_action(states)

        while decision == 0:
            # READ
            # Ignore models decision on reading when EOS in src_tokens
            if DEFAULT_EOS in states["src_tokens"]:
                break
            # Keep reading utill model decide to write
            # If there are still tokens in local buffer states["src_tokenss"],
            # don't request new raw_token from server
            if len(states["src_tokens"]) > states["src_step"] + 1 and len(states["src_tokens"]) > 0:
                states["src_step"] += 1
            elif states["src_step"] + 1 == len(states["src_tokens"]) or len(states["src_tokens"]) == 0:
                return{'key': GET, 'value': None}
            else:
                raise RuntimeError("Something is wrong." + f"{states['src_step'] + 1}, {states['src_tokens']}")
            decision = self.model.get_action(states)

        # WRITE
        tgt_token, tgt_idx = self.pred_one_target_token(states)
        states["tgt_step"] += 1 

        if tgt_idx == self.tgt_dict.eos() or self.tgt_len_from_states(states) > self.max_length(states):
            # Finish this sentence
            states["finished"] = True

        # Only send a full word to server
        if self.is_begin_of_subword(tgt_token) and self.tgt_len_from_states(states) > 1:
            raw_token = (
                self.tgt_spm
                .DecodePieces(states["tgt_subwords"][:-1])
            )
            states["tgt_tokens"] += states["tgt_subwords"][:-1]
            states["tgt_subwords"] = states["tgt_subwords"][-1:]
            states["tgt_txt"].append(raw_token) 
            return {'key': SEND, 'value': raw_token}
        else:
            return {'key': GET, 'value': None}

    def reset(self):
        pass
            