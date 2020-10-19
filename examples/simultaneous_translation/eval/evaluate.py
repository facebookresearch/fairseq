# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from agents import build_agent
from client import SimulSTEvaluationService, SimulSTLocalEvaluationService
from fairseq.registry import REGISTRIES


DEFAULT_HOSTNAME = "localhost"
DEFAULT_PORT = 12321


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hostname", type=str, default=DEFAULT_HOSTNAME, help="server hostname"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="server port number"
    )
    parser.add_argument("--agent-type", default="simul_trans_text", help="Agent type")
    parser.add_argument("--scorer-type", default="text", help="Scorer type")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index of the sentence to evaluate",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=float("inf"),
        help="End index of the sentence to evaluate",
    )
    parser.add_argument(
        "--scores", action="store_true", help="Request scores from server"
    )
    parser.add_argument("--reset-server", action="store_true", help="Reset the server")
    parser.add_argument(
        "--num-threads", type=int, default=10, help="Number of threads used by agent"
    )
    parser.add_argument(
        "--local", action="store_true", default=False, help="Local evaluation"
    )

    args, _ = parser.parse_known_args()

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    if args.local:
        session = SimulSTLocalEvaluationService(args)
    else:
        session = SimulSTEvaluationService(args.hostname, args.port)

    if args.reset_server:
        session.new_session()

    if args.agent_type is not None:
        agent = build_agent(args)
        agent.decode(session, args.start_idx, args.end_idx, args.num_threads)

    if args.scores:
        session.get_scores()
    print(session.get_scores())
