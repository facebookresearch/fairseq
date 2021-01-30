# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import sys

from scorers import build_scorer
from tornado import ioloop, web


DEFAULT_HOSTNAME = "localhost"
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
        sent_id = int(self.get_argument("sent_id"))
        segment_size = None
        if "segment_size" in self.request.arguments:
            string = self.get_argument("segment_size")
            if len(string) > 0:
                segment_size = int(string)

        r = json.dumps(self.scorer.send_src(int(sent_id), segment_size))

        self.write(r)


class HypothesisHandler(ScorerHandler):
    def put(self):
        sent_id = int(self.get_argument("sent_id"))
        list_of_tokens = self.request.body.decode("utf-8").strip().split()
        self.scorer.recv_hyp(sent_id, list_of_tokens)


def add_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                        help='Server hostname')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='Server port number')

    args, _ = parser.parse_known_args()
    # fmt: on
    return args


def start_server(scorer, hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT, debug=False):
    app = web.Application(
        [
            (r"/result", ResultHandler, dict(scorer=scorer)),
            (r"/src", SourceHandler, dict(scorer=scorer)),
            (r"/hypo", HypothesisHandler, dict(scorer=scorer)),
            (r"/", EvalSessionHandler, dict(scorer=scorer)),
        ],
        debug=debug,
    )
    app.listen(port, max_buffer_size=1024 ** 3)
    sys.stdout.write(f"Evaluation Server Started. Listening to port {port}\n")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    args = add_args()
    scorer = build_scorer(args)
    start_server(scorer, args.hostname, args.port, args.debug)
