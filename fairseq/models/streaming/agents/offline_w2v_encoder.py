import torch
from argparse import Namespace

try:
    from simuleval.agents import SpeechToSpeechAgent
    from simuleval.agents.actions import WriteAction, ReadAction, Action

    IS_SIMULEVAL_INSTALLED = True
except Exception:
    SpeechToSpeechAgent = object
    Action = WriteAction = ReadAction = object
    IS_SIMULEVAL_INSTALLED = False

from fairseq.models.speech_to_text.xm_transformer import Wav2VecEncoderWithAdaptor


class OfflineWav2VecEncoderAgent(SpeechToSpeechAgent):
    """
    Incremental encoding of an wav2vec encoder output
    It update the whole encoder states every time when there is a new incoming segment.
    """

    def __init__(self, encoder: Wav2VecEncoderWithAdaptor, args: Namespace) -> None:
        super().__init__(args)
        assert IS_SIMULEVAL_INSTALLED
        self.model = encoder
        self.model.to(self.args.device)

    @property
    def min_input_length(self):
        conv_layers = self.model.w2v_encoder.w2v_model.feature_extractor.conv_layers
        length = conv_layers[-1][0].kernel_size[0]
        for conv_layer in conv_layers:
            length *= conv_layer[0].stride[0]
        return length

    @torch.no_grad()
    def policy(self) -> Action:
        """
        The policy for encoder is always write
        only if the input is too short
        """
        if len(self.states.source) < self.min_input_length:
            if self.states.source_finished:
                return WriteAction({}, finished=self.states.source_finished)
            else:
                return ReadAction()

        torch.cuda.empty_cache()
        encoder_states = self.model(
            torch.FloatTensor(self.states.source).to(self.args.device).unsqueeze(0),
            torch.LongTensor([len(self.states.source)]).to(self.args.device),
        )

        return WriteAction(encoder_states, finished=self.states.source_finished)
