import argparse
import os.path as op
import os
import sys

from typing import List
import time
import json

from collections import defaultdict
from tornado import web, ioloop
from scorers import build_scorer

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

class ScorerHandler(web.RequestHandler):
    def initialize(self, scorer):
        self.scorer = scorer


class StartSessionHandler(ScorerHandler):
    def get(self):
        self.scorer.reset()


class EndSessionHandler(ScorerHandler):
    def get(self):
        r = json.dumps(self.scorer.score())
        self.write(r)


class GetSourceHandler(ScorerHandler):
    def get(self):
        args = self.get_argument('ids') 
        if args == "info":
            r = json.dumps(self.scorer.get_info())
        else:
            idx = int(args)
            r = json.dumps(self.scorer.send_src(idx))
        self.write(r)


class SendHypothesisHandler(ScorerHandler):
    def get(self):
        self.scorer.recv_hyp(json.loads(self.get_argument('hypo')))
        

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                        help='server hostname')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='server port number')
    parser.add_argument('--src-file', type=str,
                        help='Source file')
    parser.add_argument('--tgt-file', type=str,
                        help='Target file')
    parser.add_argument('--output', type=str,
                        help='')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    parser.add_argument('--scorer-type', type=str, default="text", choices=["text", "speech"],
                        help='Type of data to evaluate')
    parser.add_argument('--tokenizer', default="13a", choices=["none", "13a"],
                        help='Type of data to evaluate')
    args, _ = parser.parse_known_args()
    return args


def start_server(scorer, hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT, debug=False):
    app = web.Application([
        (r'/start', StartSessionHandler, dict(scorer=scorer)),
        (r'/end', EndSessionHandler, dict(scorer=scorer)),
        (r'/get', GetSourceHandler, dict(scorer=scorer)),
        (r'/send', SendHypothesisHandler, dict(scorer=scorer)),
    ], debug=debug)
    app.listen(port, max_buffer_size=1024 ** 3)
    sys.stdout.write(f"Evaluation Server Started. Listening to port {port}\n")
    ioloop.IOLoop.current().start()


if __name__ == '__main__':
    args = add_args()
    scorer = build_scorer(args)
    start_server(scorer, args.hostname, args.port, args.debug)
