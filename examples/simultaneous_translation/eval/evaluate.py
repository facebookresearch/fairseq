import argparse

import sys
from client import SimulSTEvaluationService
from agents import build_agent
from agents.registry import REGISTRIES 

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                        help='server hostname')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='server port number')
    parser.add_argument('--agent-type', required=True,
                        help='Agent type')

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
    agent = build_agent(args)
    with SimulSTEvaluationService(args.hostname, args.port) as session:
        agent.decode(session)
