import torch
from argparse import Namespace
from fairseq.models import FairseqDecoder
from fairseq.models.streaming.utils.common import test_time_waitk_agent

try:
    from simuleval.agents import GenericAgent
    from simuleval.data.segments import Segment
    from simuleval.agents.actions import Action, WriteAction, ReadAction
    from simuleval.agents.states import AgentStates

    IS_SIMULEVAL_INSTALLED = True
except Exception:
    GenericAgent = object
    AgentStates = object
    Segment = object
    Action = WriteAction = ReadAction = object
    IS_SIMULEVAL_INSTALLED = False


class FairseqDecoderAgentStates(AgentStates):
    def reset(self) -> None:
        self.incremental_states = {"steps": {"src": 0, "tgt": 1}}
        self.target_indices = []
        return super().reset()

    def update_source(self, segment: Segment):
        """
        Update states from input segment
        Additionlly update incremental states

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if not segment.is_empty:
            self.source = segment.content
            if "encoder_out" not in self.source and segment.finished:
                self.target_finished = True
                return
            self.incremental_states["steps"]["src"] = self.source["encoder_out"][
                0
            ].size(0)
            self.incremental_states["steps"]["tgt"] = 1 + len(self.target_indices)
        self.incremental_states["online"] = {"only": torch.tensor(not segment.finished)}


class OnlineTextDecoderAgent(GenericAgent):
    """
    Online text decoder
    """

    target_type = "text"

    def __init__(self, decoder: FairseqDecoder, args: Namespace) -> None:
        super().__init__(args)
        assert IS_SIMULEVAL_INSTALLED
        self.model = decoder
        self.model.to(self.args.device)
        self.model.eval()
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b

        init_target_token = getattr(args, "init_target_token", None)

        if init_target_token is None:
            init_target_token = self.model.dictionary.eos_word

        self.init_target_index = self.model.dictionary.index(init_target_token)

    def build_states(self) -> FairseqDecoderAgentStates:
        return FairseqDecoderAgentStates()

    @property
    def max_len(self):
        return (
            self.max_len_a * self.states.source["encoder_out"][0].size(0)
            + self.max_len_b
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--init-target-token",
            type=str,
            help="Initial target token used for decoding.",
        )
        parser.add_argument(
            "--max-len-a",
            type=float,
            default=0.125,
            help="Max length of predictions, a in ax + b",
        )
        parser.add_argument(
            "--max-len-b",
            type=float,
            default=10,
            help="Max length of predictions, b in ax + b",
        )

    @torch.no_grad()
    def policy(self) -> Action:
        if self.states.target_finished:
            return WriteAction("", finished=True)
        # 1. Prepare decoder input
        target_input = (
            torch.LongTensor([self.init_target_index] + self.states.target_indices)
            .unsqueeze(0)
            .to(self.args.device)
        )

        torch.cuda.empty_cache()
        decoder_states, outputs = self.model.forward(
            prev_output_tokens=target_input,
            encoder_out=self.states.source,
            incremental_state=self.states.incremental_states,
        )

        if outputs.action == 1 or self.states.source_finished:
            # If model writes
            log_probs = self.model.get_normalized_probs(
                [decoder_states[:, -1:]], log_probs=True
            )
            index = log_probs.argmax(dim=-1)[0, 0].item()

            self.states.target_indices.append(index)

            return WriteAction(
                self.model.dictionary.string([index]),
                finished=(
                    index == self.model.dictionary.eos()
                    or len(self.states.target) > self.max_len
                ),
            )

        return ReadAction()


class OnlineTextToTextDecoderAgent(OnlineTextDecoderAgent):
    source_type = "text"


class OnlineSpeechToTextDecoderAgent(OnlineTextDecoderAgent):
    source_type = "speech"


@test_time_waitk_agent
class TestTimeWaitKOnlineSpeechToTextDecoderAgent(OnlineTextDecoderAgent):
    source_type = "speech"


@test_time_waitk_agent
class TestTimeWaitKOnlineTextToTextDecoderAgent(OnlineTextDecoderAgent):
    source_type = "text"
