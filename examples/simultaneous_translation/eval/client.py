import urllib.request
import json


class SimulSTEvaluationService(object):
    DEFAULT_HOSTNAME = 'localhost'
    DEFAULT_PORT = 12321

    def __init__(self, hostname=DEFAULT_HOSTNAME, port=DEFAULT_PORT):
        self.hostname = hostname
        self.port = port
        self.base_url = f'http://{self.hostname}:{self.port}'

    def __enter__(self):
        # start eval session
        url = f'{self.base_url}/start'
        try:
            _ = urllib.request.urlopen(url)
        except Exception as e:
            print(f'Failed to start an evaluation session: {e}')
        
        print('Evaluation session started.')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # end eval session
        url = f'{self.base_url}/end'
        try:
            scores = urllib.request.urlopen(url)
        except Exception as e:
            print(f'Failed to end an evaluation session: {e}')
        
        print('Scores: {}'.format(scores.read().decode('utf-8')))
        print('Evaluation session finished.')
        

    def get_src(self) -> str:
        url = f'{self.base_url}/get'
        try:
            out = urllib.request.urlopen(url)
        except Exception as e:
            print(f'Failed to request a source segment: {e}')
        return json.loads(out.read().decode('utf-8'))

    def send_hypo(self, hypo: str) -> None:
        url = f'{self.base_url}/send?hypo={urllib.parse.quote(hypo)}'
        try:
            out = urllib.request.urlopen(url)
        except Exception as e:
            print(f'Failed to send a translated segment: {e}')





