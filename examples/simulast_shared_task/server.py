import argparse
import os.path as op
from typing import List
import time
import json


from tornado import web, ioloop

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                    help='server hostname')
parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                    help='server port number')
parser.add_argument('--data-root', type=str, default='./examples/data',
                    help='root path to data')
parser.add_argument('--debug', action='store_true', help='debug mode')
args, _ = parser.parse_known_args()


class SimulSTScorer(object):
    def __init__(self, data_root):
        self.sources = self._load_from_file(op.join(data_root, 'src.txt'))
        self.references = self._load_from_file(op.join(data_root, 'ref.txt'))
        self.translations = []
        self.timestamps = []
        self.start_timestamp = self.get_timestamp()

    @classmethod
    def get_timestamp(cls):
        return time.time()

    @classmethod
    def _load_from_file(cls, path: str) -> List[str]:
        with open(path) as f:
            return [r.strip() for r in f]

    def reset(self):
        self.sources, self.references = [], []
        self.translations, self.timestamps = [], []
        self.start_timestamp = self.get_timestamp()

    def score(self):
        from vizseq.scorers.bleu import BLEUScorer
        from vizseq.scorers.ter import TERScorer
        from vizseq.scorers.meteor import METEORScorer
        # TODO: implement latency metrics
        bleu_score = BLEUScorer(sent_level=False, corpus_level=True).score(
            self.translations, [self.references]
        )
        ter_score = TERScorer(sent_level=False, corpus_level=True).score(
            self.translations, [self.references]
        )
        meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
            self.translations, [self.references]
        )
        return {'BLEU': bleu_score, 'TER': ter_score, 'METEOR': meteor_score}

    def add_example(self, translation: str) -> None:
        self.translations.append(translation)
        self.timestamps.append(self.get_timestamp())


scorer = SimulSTScorer(args.data_root)


class StartSessionHandler(web.RequestHandler):
    def get(self):
        scorer.reset()


class EndSessionHandler(web.RequestHandler):
    def get(self):
        r = json.dumps(scorer.score())
        self.write(r)


class GetSourceHandler(web.RequestHandler):
    def get(self):
        raise NotImplementedError


class SendHypothesisHandler(web.RequestHandler):
    def post(self):
        hypo = ''
        scorer.add_example(hypo)
        raise NotImplementedError


def start_server(hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT, debug=False):
    app = web.Application([
        (r'/start', StartSessionHandler),
        (r'/end', EndSessionHandler),
        (r'/get', GetSourceHandler),
        (r'/send', SendHypothesisHandler),
    ], debug=debug)
    app.listen(port, max_buffer_size=1024 ** 3)
    print("Evaluation Server Started")
    ioloop.IOLoop.current().start()


if __name__ == '__main__':
    start_server(args.hostname, args.port, args.debug)
