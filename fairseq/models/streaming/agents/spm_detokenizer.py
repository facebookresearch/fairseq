from argparse import ArgumentParser
from fairseq.data.encoders import build_bpe

try:
    from simuleval.agents import TextToTextAgent
    from simuleval.agents.actions import WriteAction, ReadAction

    IS_SIMULEVAL_INSTALLED = True
except:
    TextToTextAgent = object
    WriteAction = ReadAction = object
    IS_SIMULEVAL_INSTALLED = False


class SentencePieceModelDetokenizerAgent(TextToTextAgent):
    def __init__(self, args):
        super().__init__(args)
        assert IS_SIMULEVAL_INSTALLED
        self.args.bpe = "sentencepiece"
        spm_processor = build_bpe(self.args)
        self.spm_processor = spm_processor

    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--sentencepiece-model",
            type=str,
            help="Path to sentencepiece model.",
            required=True,
        )

    def policy(self):

        possible_full_words = self.spm_processor.decode(
            " ".join([x for x in self.states.source])
        )

        if self.states.source_finished:
            return WriteAction(possible_full_words, True)
        elif len(possible_full_words.split()) > 1:
            full_word = possible_full_words.split()[0]
            self.states.source = self.states.source[-1:]
            return WriteAction(full_word, finished=False)
        else:
            return ReadAction()
