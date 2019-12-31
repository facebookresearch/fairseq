import argparse
import os.path as op
import os

from typing import List
import time
import json


from tornado import web, ioloop
from examples.simultaneous_translation.data.data_loader import TextDataLoader
from examples.simultaneous_translation.utils.eval_latency import LatencyScorer

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321
DEFAULT_EOS = '</s>'

parser = argparse.ArgumentParser()
parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                    help='server hostname')
parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                    help='server port number')
#parser.add_argument('--data-root', type=str, default='./examples/data',
#                    help='root path to data')
parser.add_argument('--src-file', type=str,
                    help='Source file')
parser.add_argument('--ref-file', type=str,
                    help='Target file')
parser.add_argument('--debug', action='store_true', help='debug mode')
args, _ = parser.parse_known_args()


class SimulSTScorer(object):
    def __init__(self, src_file, ref_file):
        self.data = {"src" : src_file, "ref" : ref_file}
        self.references = self._load_from_file(ref_file)
        self.reset()
        
    @classmethod
    def _load_from_file(cls, file):
        with open(file) as f:
            return [r.strip() for r in f]

    def send_src(self):
        try:
            # Unlock to read the next sentence
            if self.sources is None:
                if self.curr_step == -1: # unlock reading
                    self.sources = next(self.source_loader)
                else:
                    return {}
                            
            sent_id, line = self.sources
            self.curr_step = self.curr_step + 1
            if self.curr_step < len(line):
                segment = line[self.curr_step]
            else:
                segment, self.sources = DEFAULT_EOS, None
            
            return {'sent_id': sent_id, 'segment_id': self.curr_step, 'segment': segment}

        except StopIteration:
            return {}

    def recv_hyp(self, hypo):
        self.translations[-1].append((hypo, self.curr_step))
        if hypo == DEFAULT_EOS:
            self.sources, self.curr_step = None, -1
            self.translations.append([])

    def reset(self):
        self.source_loader = TextDataLoader(self.data["src"]).load()
        self.sources = None
        self.translations = [[]]
        self.curr_step = -1
    
    def src_lengths(self):
        return [len(item[1]) for item in TextDataLoader(self.data["src"]).load()]


    def score(self):
        from vizseq.scorers.bleu import BLEUScorer
        from vizseq.scorers.ter import TERScorer
        from vizseq.scorers.meteor import METEORScorer

        translations = [" ".join([ti[0] for ti in t[:-1]]) for t in self.translations[:-1]]

        # TODO: implement latency metrics
        bleu_score = BLEUScorer(sent_level=False, corpus_level=True).score(
            translations, [self.references]
        )
        ter_score = TERScorer(sent_level=False, corpus_level=True).score(
            translations, [self.references]
        )
        meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
            translations, [self.references]
        )

        delays = [[ti[1] for ti in t[:-1]] for t in self.translations[:-1]]

        src_lengths = [len(item[1]) for item in TextDataLoader(self.data["src"]).load()]

        latency_score = LatencyScorer().score(
            [{"src_len" : src_len, "delays" : delay} for src_len, delay in zip(src_lengths, delays)]
        )

        return {
            'BLEU': bleu_score[0], 
            'TER': ter_score[0], 
            'METEOR': meteor_score[0],
            'DAL' : latency_score['differentiable_average_lagging'],
            'AL' : latency_score['average_lagging'],
            'AP' : latency_score['average_proportion'],
        }


scorer = SimulSTScorer(args.src_file, args.ref_file)


class StartSessionHandler(web.RequestHandler):
    def get(self):
        scorer.reset()


class EndSessionHandler(web.RequestHandler):
    def get(self):
        r = json.dumps(scorer.score())
        self.write(r)


class GetSourceHandler(web.RequestHandler):
    def get(self):
        r = json.dumps(scorer.send_src())
        self.write(r)


class SendHypothesisHandler(web.RequestHandler):
    def get(self):
        scorer.recv_hyp(self.get_argument('hypo', 'no hypothesis received'))
        

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
