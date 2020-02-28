
from . simul_trans_agent import SimulTransAgent
from . import DEFAULT_EOS, GET, SEND
from . import register_agent
from .word_splitter import *
import os


@register_agent("simul_trans_speech")
class SimulTransSpeechAgent(SimulTransAgent):
    def __init__(self, args):
        super().__init__(args)
        
        # TODO: Change to configurable params
        self.num_mel_bins = 40 
        self.frame_length = 25
        self.frame_shift = 10
        self.sample_rate = 16000
    
    def init_states(self):
        states = super().init_states()
        states["frame_length"] = self.frame_shift
        states["finish_read"] = False
        return states

    def build_word_splitter(self, args):
        self.word_splitter = {}

        self.word_splitter["tgt"] = eval(f"{args.tgt_splitter_type}WordSplitter")(
                getattr(args, f"tgt_splitter_path")
            )

    def load_dictionary(self, task):
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

    def update_states(self, states, new_state):

        # utterence is a list
        # len = sample_rate / 1000 * segment_size (ms)
        # When sample_rate = 16000h, segment_size = 10ms
        # len = 160
        utterence = new_state["segment"]

        if utterence not in [self.eos]:
            # TODO: make it incremantal
            states['segments']['src'] += utterence

            if (
                len(states['segments']['src']) 
                >= self.sample_rate / 1000 * self.frame_length
            ):
                import torch
                import torchaudio.compliance.kaldi as kaldi
                from examples.simultaneous_translation.data.data_utils import apply_mv_norm
                torch.manual_seed(0)
                output = kaldi.fbank(
                    torch.FloatTensor(states["segments"]["src"]).unsqueeze(0),
                    num_mel_bins=self.num_mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift
                )

                # TODO: apply_mv_norm function calculation mean and var along the time axis
                # This caused a mismatch between the train and inference
                states["indices"]['src'] = apply_mv_norm(output)

            states["steps"]["src"] += len(utterence) / self.sample_rate * 1000
        else:
            states["finish_read"] = True

        return states

    def read_action(self, states):
        segment_size = self.model.decoder.attention.segment_size(
            self.frame_shift,
            self.model.subsampling_factor()
        )
        return {'key': GET, 'value': {"segment_size": segment_size}}

    def finish_read(self, states):
        return states["finish_read"]
