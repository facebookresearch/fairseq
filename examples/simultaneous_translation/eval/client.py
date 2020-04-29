# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
from typing import Optional
from scorers import build_scorer


class SimulSTEvaluationService(object):
    DEFAULT_HOSTNAME = 'localhost'
    DEFAULT_PORT = 12321

    def __init__(self, hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT):
        self.hostname = hostname
        self.port = port
        self.base_url = f'http://{self.hostname}:{self.port}'

    def __enter__(self):
        self.new_session()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def new_session(self):
        # start eval session
        url = f'{self.base_url}'

        try:
            _ = requests.post(url)
        except Exception as e:
            print(f'Failed to start an evaluation session: {e}')

        print('Evaluation session started.')
        return self

    def get_scores(self):
        # end eval session
        url = f'{self.base_url}/result'
        try:
            r = requests.get(url)
            print('Scores: {}'.format(r.json()))
            print('Evaluation session finished.')
        except Exception as e:
            print(f'Failed to end an evaluation session: {e}')

    def get_src(self, sent_id: int, extra_params: Optional[dict] = None) -> str:
        url = f'{self.base_url}/src'
        params = {"sent_id": sent_id}
        if extra_params is not None:
            for key in extra_params.keys():
                params[key] = extra_params[key]
        try:
            r = requests.get(
                url,
                params=params
            )
        except Exception as e:
            print(f'Failed to request a source segment: {e}')
        return r.json()

    def send_hypo(self, sent_id: int, hypo: str) -> None:
        url = f'{self.base_url}/hypo'
        params = {"sent_id": sent_id}

        try:
            requests.put(url, params=params, data=hypo.encode("utf-8"))
        except Exception as e:
            print(f'Failed to send a translated segment: {e}')

    def corpus_info(self):
        url = f'{self.base_url}'
        try:
            r = requests.get(url)
        except Exception as e:
            print(f'Failed to request corpus information: {e}')

        return r.json()


class SimulSTLocalEvaluationService(object):
    def __init__(self, args):
        self.scorer = build_scorer(args)

    def get_scores(self):
        return self.scorer.score()

    def get_src(self, sent_id: int, extra_params: Optional[dict] = None) -> str:
        if extra_params is not None:
            segment_size = extra_params.get("segment_size", None)
        else:
            segment_size = None

        return self.scorer.send_src(int(sent_id), segment_size)

    def send_hypo(self, sent_id: int, hypo: str) -> None:
        list_of_tokens = hypo.strip().split()
        self.scorer.recv_hyp(sent_id, list_of_tokens)

    def corpus_info(self):
        return self.scorer.get_info()
