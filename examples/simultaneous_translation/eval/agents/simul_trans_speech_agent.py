from .simul_trans_text_agent import SimulTransTextAgent
from . import register_agent
from . import DEFAULT_EOS, GET, SEND
from . import register_agent
import torch
from fairseq import checkpoint_utils, options, progress_bar, utils, tasks
from .word_splitter import *
import os
import torchaudio.compliance.kaldi as kaldi

@register_agent("simul_trans_speech")
class SimulTransSpeechAgent(SimulTransTextAgent):
    def __init__(self, args):
        self.word_splitter = {}
        self.word_splitter["tgt"] = eval(f"{args.tgt_splitter_type}WordSplitter")(
                getattr(args, f"tgt_splitter_path")
            )

        # Load Model
        self.load_model(args)

        self.max_len = args.max_len

        # Queue to store speech signal
        self.num_mel_bins = 40 
        self.frame_length = 25
        self.frame_shift = 10
        self.audio_queue_size = (self.frame_length + self.frame_shift - 1) // self.frame_shift
        self.audio_queue = []

    def load_model(self, args):
        args.user_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        utils.import_user_module(args)
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename, eval(args.model_overrides))

        args = state["args"]
        args.task = "speech_translation"

        task = tasks.setup_task(args)

        # build model for ensemble
        self.model = task.build_model(args)
        self.model.load_state_dict(state["model"], strict=True)
        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def _push_audio_queue(self, audio):
        if len(self.audio_queue) < self.audio_queue_size:
            self.audio_queue.append(audio)
            return

        if audio is None:
            self.audio_queue = self.audio_queue[1:]

        self.audio_queue = self.audio_queue[1:] + [audio]

    def update_states(self, states, new_state):
        if len(new_state) == 0:
            return new_state 

        utterence = new_state["segment"]
        states["words"]["src"] += utterence

        if utterence not in [DEFAULT_EOS]:
            audio = torch.FloatTensor(new_state['segment']).unsqueeze(0)
            self._push_audio_queue(audio)
            if len(self.audio_queue) == self.audio_queue_size:
                import pdb; pdb.set_trace()
                features = kaldi.fbank(
                    torch.cat(self.audio_queue, dim=1),
                    num_mel_bins=self.num_mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift
                ).unsqueeze(1)
                states["indices"]["src"] += features
                states["tokens"]["src"] += audio
            else:
                audio = None
                features = None
        else:
            audio = DEFAULT_EOS
            features = torch.FloatTensor()

        # Update states

        return states

    def read_action(self, states):
        states["steps"]["src"] += 1
        return {'key': GET, 'value': None}

