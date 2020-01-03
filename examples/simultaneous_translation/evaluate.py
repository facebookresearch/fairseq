import argparse

from examples.simultaneous_translation.eval.client import SimulSTEvaluationService
from examples.simultaneous_translation.eval.agent import DummyWaitAgent, SimulTransAgentBuilder

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321

parser = argparse.ArgumentParser()

parser.add_argument('--hostname', type=str, default=DEFAULT_HOSTNAME,
                    help='server hostname')
parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                    help='server port number')
parser.add_argument('--data-type', type=str, default="text",
                    help='Type of data to evaluate')

agent_builder = SimulTransAgentBuilder() 

parser = agent_builder.add_args(parser)
args = parser.parse_args()

agent = agent_builder(args)

with SimulSTEvaluationService(args.hostname, args.port) as session:
    agent.decode(session)
