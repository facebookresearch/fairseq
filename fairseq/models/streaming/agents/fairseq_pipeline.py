from argparse import ArgumentParser

try:
    from simuleval.agents import AgentPipeline
    from simuleval.data.segments import Segment

    IS_SIMULEVAL_INSTALLED = True
except Exception:
    AgentPipeline = object
    Segment = None
    IS_SIMULEVAL_INSTALLED = False


class FairseqAgentPipeline(AgentPipeline):
    pipeline = []

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to the model checkpoint.",
        )
        parser.add_argument(
            "--device", type=str, default="cuda:0", help="Device to run the model."
        )
        parser.add_argument(
            "--config-yaml", type=str, default=None, help="Path to config yaml"
        )
    
    def pop(self) -> Segment:
        output_segment = super().pop()

        if not self.module_list[0].states.source_finished and output_segment.finished:
            # An early stop.
            # The temporary solution is to start over
            self.reset()
            output_segment.finished = False

        return output_segment

    @classmethod
    def from_args(cls, args):
        return cls(args)
