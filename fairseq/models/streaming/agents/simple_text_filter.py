from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from argparse import ArgumentParser


class SimpleTextFilterAgent(TextToTextAgent):
    """
    Simply filter out the some tokens in the output.
    """

    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--filtered-tokens",
            nargs="+",
            type=str,
            help="List of tokens need to be filtered.",
            required=True,
        )

    def policy(self) -> Action:
        if len(self.states.source) == 0:
            if self.states.source_finished:
                return WriteAction("", finished=True)
            return ReadAction()

        if self.states.source[-1] in self.args.filtered_tokens:
            return ReadAction()
        else:
            token = self.states.source[-1]
            self.states.source = []
            return WriteAction(token, finished=self.states.source_finished)
