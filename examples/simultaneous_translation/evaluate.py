import argparse

from examples.simultaneous_translation.eval.client import SimulSTEvaluationService
from examples.simultaneous_translation.eval.agent import DummyWaitAgent, SimulTransAgent

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

parser = SimulTransAgent.argument_parser()

parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                    help='server hostname')
parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                    help='server port number')
parser.add_argument('--model-path', type=str, default=None, 
                    help='path to your pretrained model.')

agent = SimulTransAgent(parser)

args = agent.args

with SimulSTEvaluationService(args.hostname, args.port) as session:
    agent.decode(session)
