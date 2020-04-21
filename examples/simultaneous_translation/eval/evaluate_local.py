import argparse

import sys
from client import SimulSTEvaluationService, SimulSTLocalEvaluationService
from agents import build_agent
from agents.registry import REGISTRIES
from scorers import build_scorer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent-type', default=None,
                        help='Agent type')
    parser.add_argument('--scorer-type', type=str, default="text", choices=["text", "speech"],
                        help='Type of data to evaluate')
    parser.add_argument('--num-threads', type=int, default=10,
                        help='Number of threads used by agent')
    parser.add_argument('--src-file', type=str,
                        help='Source file')
    parser.add_argument('--tgt-file', type=str,
                        help='Target file')
    parser.add_argument('--output', type=str,
                        help='Directory to leave output')
    parser.add_argument('--tokenizer', default="13a", choices=["none", "13a"],
                        help='Type of data to evaluate')
    parser.add_argument('--tgt-file-type', type=str, default="json",
                        choices=['json', "text"], required=False,
                        help='Type of the tgt_file, choose from json, text')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args, _ = parser.parse_known_args()
    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY['registry'][choice]
            if hasattr(cls, 'add_args'):
                cls.add_args(parser)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    scorer = build_scorer(args)
    session = SimulSTLocalEvaluationService(scorer)

    if args.agent_type is not None:
        agent = build_agent(args)
        agent.decode(session, num_thread=args.num_threads)

    print(session.get_scores())
