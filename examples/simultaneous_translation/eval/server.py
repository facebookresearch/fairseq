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


class EvalSessionHandler(ScorerHandler):
    def post(self):
        self.scorer.reset()

    def get(self):
        r = json.dumps(self.scorer.get_info())
        self.write(r)


class ResultHandler(ScorerHandler):
    def get(self):
        r = json.dumps(self.scorer.score())
        self.write(r)


class SourceHandler(ScorerHandler):
    def get(self):
        sent_id = int(self.get_argument('sent_id'))
        segment_size = None
        if "segment_size" in self.request.arguments:
            string = self.get_argument('segment_size')
            if len(string) > 0:
                segment_size = int(string)
            

        r = json.dumps(
            self.scorer.send_src(
                int(sent_id),
                segment_size
            )
        )
        self.write(r)


class HypothesisHandler(ScorerHandler):
    def put(self):
        sent_id = int(self.get_argument('sent_id'))
        list_of_tokens = (
            self.request.body
            .decode('utf-8')
            .strip()
            .split()
        )
        self.scorer.recv_hyp(sent_id, list_of_tokens)


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
    parser.add_argument('--tgt-file-type', type=str, default="json",
                        choices=['json', "text"], required=False,
                        help='Type of the tgt_file, choose from json, text')
    args, _ = parser.parse_known_args()
    return args


def start_server(scorer, hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT, debug=False):
    app = web.Application([
        (r'/result', ResultHandler, dict(scorer=scorer)),
        (r'/src', SourceHandler, dict(scorer=scorer)),
        (r'/hypo', HypothesisHandler, dict(scorer=scorer)),
        (r'/', EvalSessionHandler, dict(scorer=scorer)),
    ], debug=debug)
    app.listen(port, max_buffer_size=1024 ** 3)
    sys.stdout.write(f"Evaluation Server Started. Listening to port {port}\n")
    ioloop.IOLoop.current().start()


if __name__ == '__main__':
    args = add_args()
    scorer = build_scorer(args)
    start_server(scorer, args.hostname, args.port, args.debug)
