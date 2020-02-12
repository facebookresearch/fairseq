from . import GET, SEND, DEFAULT_EOS
class Agent(object):
    "an agent needs to follow this pattern"
    def __init__(self, *args, **kwargs):
        pass
    def init_states(self):
        raise NotImplementedError

    def update_states(self, states, new_state):
        raise NotImplementedError

    def finish_eval(self, states, new_state):
        raise NotImplementedError

    def policy(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
        
    def decode(self, session):
        states = self.init_states()
        self.reset()
        
        while True:
            # take an action
            action = self.policy(states)
            if action['key'] == GET:
                new_state = session.get_src()
                if self.finish_eval(states, new_state):
                    break # end of document
                states = self.update_states(states, new_state)

            elif action['key'] == SEND:
                session.send_hypo(action['value'])
                if action['value'] == DEFAULT_EOS:
                    states = self.init_states()  # clean the history, wait for next sentence
                    self.reset()
            else:
                raise NotImplementedError
