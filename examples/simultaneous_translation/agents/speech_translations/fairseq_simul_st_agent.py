import os
import torch
import math
import yaml
import pickle
from simuleval.agents import SpeechAgent
from simuleval.states import ListEntry, SignalEntry
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS

import torchaudio.compliance.kaldi as kaldi
import numpy as np

from fairseq import checkpoint_utils, tasks

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80


class OnlineFeatureExtractor:
    def __init__(
        self,
        shift_size=SHIFT_SIZE,
        window_size=WINDOW_SIZE,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        global_cmvn={"mean": 0, "std": 1},
    ):
        self.shift_size = shift_size
        self.window_size = window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.num_samples_per_shift = int(SHIFT_SIZE * SAMPLE_RATE / 1000)
        self.num_samples_per_window = int(WINDOW_SIZE * SAMPLE_RATE / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        # TODO: Make it transform later
        self.global_cmvn = global_cmvn

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        num_frames = math.floor(
            (
                len(samples)
                - self.len_ms_to_samples(self.window_size - self.shift_size)
            )
            / self.num_samples_per_shift
        )

        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift:
        ]

        torch.manual_seed(1)
        x = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        ).numpy()

        x = np.subtract(x, self.global_cmvn["mean"])
        x = np.divide(x, self.global_cmvn["std"])

        return torch.from_numpy(x)


class FairseqSimulSTAgent(SpeechAgent):
    speech_segment_size = 40

    def __init__(self, args):
        super().__init__(args)
        # Initialize your agent here, for example load model, vocab, etc

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, 'gpu', False)

        self.args = args

        self.load_model_vocab(args)

        config_yaml = os.path.join(args.data_bin, 'config.yaml')
        with open(config_yaml, 'r') as f:
            config = yaml.safe_load(f)

        if 'global_cmvn' in config:
            with open(config['global_cmvn']['gcmvn'], 'rb') as f:
                gcmvn = pickle.load(f)
        else:
            gcmvn = {"mean": 0, "std": 1}

        self.feature_extractor = OnlineFeatureExtractor(
            global_cmvn=gcmvn
        )

        self.max_len = args.max_len

        self.force_finish_read = args.force_finish_read

        torch.set_grad_enabled(False)

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--tgt-splitter-type", type=str,
                            default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str,
                            default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--force-finish-read", default=False,
                            action="store_true",
                            help=(
                                "Force the model to finish read the whole"
                                " input even the model predict eos"
                            ))
        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        saved_args = state["args"]
        saved_args.data = args.data_bin

        task = tasks.setup_task(saved_args)

        # build model for ensemble
        self.model = task.build_model(saved_args)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = SignalEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()

    def segment_to_units(self, segment, states):
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # TODO: Refactor here to make it more readable
        if self.model.decoder.dictionary.eos() == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.model.decoder.dictionary.string([index])
            if token.startswith("\u2581"):
                if len(segment) == 0:
                    segment += [token.replace("\u2581", "")]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace("\u2581", "")]

        if (
            len(units) > 0
            and (
                self.model.decoder.dictionary.eos() == units[-1]
                or len(states.units.target) >= self.max_len
            )
        ):
            tokens = [
                self.model.decoder.dictionary.string([unit])
                for unit in units
            ]
            return [''.join(tokens).replace('\u2581', '')] + [DEFAULT_EOS]

        return None

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(states.units.source.value.unsqueeze(0))
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action
        self.update_model_encoder(states)

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [x for x in states.units.target.value if x is not None]
            ).unsqueeze(0)
        )

        states.incremental_states["steps"] = {
            "src": states.encoder_states.encoder_out.size(0),
            "tgt": 1 + len(states.units.target),
        }

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states.encoder_out,
            incremental_state=states.incremental_states,
        )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        if outputs["action"] == 0:
            return READ_ACTION
        else:
            return WRITE_ACTION

    def predict(self, states):

        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]],
            log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        torch.cuda.empty_cache()

        index = index[0, 0].item()

        if (
            self.force_finish_read
            and index == self.model.decoder.dictionary.eos()
            and not states.finish_read()
        ):
            index = None

        return index
