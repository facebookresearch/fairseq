
from . simul_trans_text_agent import SimulTransTextAgent
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
        
        # TODO: Change to configurable params
        self.num_mel_bins = 40 
        self.frame_length = 25
        self.frame_shift = 10
        self.sample_rate = 16000
        self.block_size = 10

    def load_dictionary(self, task):
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def update_states(self, states, new_state):

        # utterence is a list
        # len = sample_rate / 1000 * blocK_size (ms)
        # When sample_rate = 16000h, block_size = 10ms
        # len = 160
        utterence = new_state["segment"]
        if utterence not in [DEFAULT_EOS]:
            # TODO: make it incremantal
            states['segments']['src'] += utterence

            if (
                len(states['segments']['src']) 
                >= self.sample_rate / 1000 * self.frame_length
            ):
                torch.manual_seed(0)
                states["indices"]['src'] = kaldi.fbank(
                    torch.FloatTensor(states["segments"]["src"]).unsqueeze(0),
                    num_mel_bins=self.num_mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift
                )

        return states
    
