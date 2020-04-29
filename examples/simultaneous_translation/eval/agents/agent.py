# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import GET, SEND, DEFAULT_EOS
import time
from multiprocessing.pool import ThreadPool as Pool
from functools import partial


class Agent(object):
    "an agent needs to follow this pattern"
    def __init__(self, *args, **kwargs):
        pass

    def init_states(self, *args, **kwargs):
        raise NotImplementedError

    def update_states(self, states, new_state):
        raise NotImplementedError

    def finish_eval(self, states, new_state):
        raise NotImplementedError

    def policy(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def decode(self, session, low=0, high=100000, num_thread=10):
        corpus_info = session.corpus_info()
        high = min(corpus_info["num_sentences"] - 1, high)
        if low >= high:
            return

        t0 = time.time()
        if num_thread > 1:
            with Pool(10) as p:
                p.map(
                    partial(self._decode_one, session),
                    [sent_id for sent_id in range(low, high + 1)]
                )
        else:
            for sent_id in range(low, high + 1):
                self._decode_one(session, sent_id)

        print(f'Finished {low} to {high} in {time.time() - t0}s')

    def _decode_one(self, session, sent_id):
        action = {}
        self.reset()
        states = self.init_states()
        while action.get('value', None) != DEFAULT_EOS:
            # take an action
            action = self.policy(states)

            if action['key'] == GET:
                new_states = session.get_src(sent_id, action["value"])
                states = self.update_states(states, new_states)

            elif action['key'] == SEND:
                session.send_hypo(sent_id, action['value'])
        print(" ".join(states["tokens"]["tgt"]))
