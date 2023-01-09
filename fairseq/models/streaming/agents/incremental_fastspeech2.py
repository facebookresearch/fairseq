import torch
from typing import Tuple, List
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

try:
    from simuleval.agents import TextToSpeechAgent
    from simuleval.agents.actions import WriteAction, ReadAction
    from simuleval.data.segments import SpeechSegment

    IS_SIMULEVAL_INSTALLED = True
except Exception:
    TextToSpeechAgent = object
    WriteAction = ReadAction = SpeechSegment = object
    IS_SIMULEVAL_INSTALLED = False

try:
    from g2p_en import G2p

    IS_G2P_INSTALLED = True
except:
    IS_G2P_INSTALLED = False


class IncrementalFastspeech2(TextToSpeechAgent):
    """
    Incrementally feed text to this offline Fastspeech2 TTS model,
    with a minimum numbers of phonemes every chunk.
    """

    def __init__(self, args):
        super().__init__(args)
        assert IS_SIMULEVAL_INSTALLED
        assert IS_G2P_INSTALLED
        self.min_phoneme = args.min_phoneme
        self.device = torch.device(args.device)
        self.load_tts()
        self.g2p = G2p()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--min-phoneme", default=6, type=int)

    def load_tts(self):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)
        self.tts_task = task
        self.tts_models = [model.to(self.device) for model in models]

    def compute_phoneme_count(self, string: str) -> int:
        return len([x for x in self.g2p(string) if x != " "])

    def get_tts_output(self, text: str) -> Tuple[List[float], int]:
        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        if sample["net_input"]["src_lengths"][0] == 0:
            return [], 0
        for key in sample["net_input"].keys():
            if sample["net_input"][key] is not None:
                sample["net_input"][key] = sample["net_input"][key].to(self.device)

            torch.cuda.empty_cache()
            wav, rate = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_models[0], self.tts_generator, sample
            )
            wav = wav.cpu().tolist()
            return wav, rate

    def policy(self) -> None:
        current_phoneme_counts = self.compute_phoneme_count(
            " ".join(self.states.source)
        )
        if current_phoneme_counts >= self.min_phoneme or self.states.source_finished:
            samples, fs = self.get_tts_output(self.states.source)

            if not self.states.source_finished:
                self.reset()

            return WriteAction(
                SpeechSegment(
                    content=samples,
                    sample_rate=fs,
                    finished=self.states.source_finished,
                ),
                finished=self.states.source_finished,
            )
        return ReadAction()
