DEFAULT_EOS = '</s>'
GET = 0
SEND = 1
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
import torchaudio.compliance.kaldi as kaldi
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


class SimulTransAgentBuilder(Agent):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model-path', type=str, default=None, 
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--user-dir", type=str,
                            help="User directory for simultaneous translation")
        parser.add_argument("--src-spm", type=str,
                            help="Source side sentence piece model")
        parser.add_argument("--tgt-spm", type=str,
                            help="Target side sentence piece model")
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

    def __call__(self, args):
        args.data = getattr(args, "data", args.data_bin)
        if args.data_type == "text":
            args.task = "translation"
            return SimulTextTransAgent(args)
        elif args.data_type == "speech":
            args.task = "speech_translation"
            return SimulSpeechTransAgent(args)
        else:
            raise NotImplementedError


class SimulTextTransAgent(Agent):
    def __init__(self, args):
        self.use_cuda = torch.cuda.is_available() and not args.cpu

        # Load Model
        _, task = self.load_model(
            args.model_path,
            arg_overrides=eval(args.model_overrides),  # noqa
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
        return args, task

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
        #print(states["src_indices"])
        #print(new_state_info)
        #import pdb; pdb.set_trace()

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
        src_tokens = states["src_indices"][:, :1 + states["src_step"]]
        src_lengths = torch.LongTensor([src_tokens.size(1)])

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
            return self.finish_action()
        
        # Model make decision given current states
        decision = self.model.get_action(states)

        if decision == 0:
            # READ
            action = self.read_action(states)
            if action is not None:
                return action
        # WRITE 
        return self.write_action(states)

    def write_action(self, states):
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

    def read_action(self, states):
        decision = 0
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
                return {'key': GET, 'value': None}
            else:
                raise RuntimeError("Something is wrong." + f"{states['src_step'] + 1}, {states['src_tokens']}")
            decision = self.model.get_action(states)

        return None

    def finish_action(self):
        return {'key': SEND, 'value': DEFAULT_EOS}

    def reset(self):
        pass


class SimulSpeechTransAgent(SimulTextTransAgent):
    def __init__(self, args):
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        task = tasks.setup_task(args) 
        # Load Model
        state_args, _ = self.load_model(
            args.model_path,
            arg_overrides=eval(args.model_overrides),  # noqa
            task=task
        )
        #self.num_mel_bins = state_args.num_mel_bins
        #self.frame_length = state_args.frame_length
        #self.frame_shift = state_args.frame_shift
        self.num_mel_bins = 40 
        self.frame_length = 25
        self.frame_shift = 10
        # Set dictionary
        self.tgt_dict = task.target_dictionary

        # Load SPM model
        self.tgt_spm = spm.SentencePieceProcessor()
        self.tgt_spm.Load(args.tgt_spm)

        self.max_len = args.max_len

        # Queue to store speech signal
        self.audio_queue_size = (self.frame_length + self.frame_shift - 1) // self.frame_shift
        self.audio_queue = []
    
    def push(self, audio):
        if len(self.audio_queue) < self.audio_queue_size:
            self.audio_queue.append(audio)
            return

        if audio is None:
            self.audio_queue = self.audio_queue[1:]

        self.audio_queue = self.audio_queue[1:] + [audio]

    def prepocess_state(self, state):
        if len(state) == 0:
            return state 
        
        utterence = state["segment"]

        if utterence not in [DEFAULT_EOS]:
            audio = torch.FloatTensor(state['segment']).unsqueeze(0)
            self.push(audio)
            if len(self.audio_queue) == self.audio_queue_size:
                features = kaldi.fbank(
                    torch.cat(self.audio_queue, dim=1),
                    num_mel_bins=self.num_mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift
                ).unsqueeze(1)
            else:
                audio = None
                features = None

            if self.use_cuda:
                features.cuda()
        else:
            audio = DEFAULT_EOS
            features = torch.FloatTensor()
        
        return {
            "src_txt": audio, 
            "src_tokens": [features], 
            "src_indices": features, 
            "sent_id" : state["sent_id"],
            "segment_id" : state["segment_id"]
        }

    def read_action(self, states):
        if len(states["src_txt"]) > 0 and type(states["src_txt"][-1]) == str and states["src_txt"][-1] == DEFAULT_EOS:
            return
        states["src_step"] += 1
        return {'key': GET, 'value': None}


    def reset(self):
        pass
            