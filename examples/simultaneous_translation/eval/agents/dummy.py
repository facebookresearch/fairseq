from .agent import Agent
from . import GET, SEND

class DummyWaitAgent(Agent):
    def __init__(self, k=2, *args, **kwargs):
        self.wait_k = k
        self.curr_k = 0

    def init_states(self):
        raise []


    def update_states(self, states, new_state):
        return states + [new_state]

    def finish_eval(self, states, new_state):
        if len(new_state) == 0 and len(states) == 0:
            return True
        return False

    def policy(self, states):
        if len(states) - self.curr_k < self.wait_k:
            action = {'key': GET, 'value': None}
        else:
            action = {'key': SEND, 'value': states[self.curr_k]['segment']}
            self.curr_k += 1
        return action

    def reset(self):
        self.curr_k = 0
