import json
import torch
from argparse import ArgumentParser
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

try:
    from simuleval.agents import TextToSpeechAgent
    from simuleval.agents.actions import WriteAction, ReadAction
    from simuleval.data.segments import SpeechSegment

    IS_SIMULEVAL_IMPORTED = True
except Exception:
    TextToSpeechAgent = object
    WriteAction = ReadAction = object
    SpeechSegment = object
    IS_SIMULEVAL_IMPORTED = False


class CodeHiFiGANVocoderAgent(TextToSpeechAgent):
    def __init__(self, args):
        super().__init__(args)
        assert IS_SIMULEVAL_IMPORTED
        self.device = args.device
        self.fs = 16000
        self.dur_prediction = args.dur_prediction
        self.device = torch.device(args.device)
        with open(args.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
            self.vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
            self.vocoder.to(self.device)

    def policy(self) -> WriteAction:
        """
        The policy is always write if there are units
        """

        code_input = []
        for segment in self.states.source:
            code_input += [int(x) for x in segment.split()]

        if len(code_input) == 0:
            if self.states.source_finished:
                return WriteAction([], finished=True)
            else:
                return ReadAction()

        x = {
            "code": torch.LongTensor(code_input).view(1, -1).to(self.device),
        }

        torch.cuda.empty_cache()
        wav_samples = self.vocoder(x, self.dur_prediction).cpu().tolist()
        self.states.source = []

        return WriteAction(
            SpeechSegment(
                content=wav_samples,
                finished=self.states.source_finished,
                sample_rate=self.fs,
            ),
            finished=self.states.source_finished,
        )

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--vocoder",
            type=str,
            required=True,
            help="path to the CodeHiFiGAN vocoder checkpoint",
        )
        parser.add_argument(
            "--vocoder-cfg",
            type=str,
            required=True,
            help="path to the CodeHiFiGAN vocoder config",
        )
        parser.add_argument(
            "--dur-prediction",
            action="store_true",
            help="enable duration prediction (for reduced/unique code sequences)",
        )
