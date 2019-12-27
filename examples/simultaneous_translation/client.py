import urllib.request


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

    def __exit__(self, exc_type, exc_val, exc_tb):
        # end eval session
        url = f'{self.base_url}/end'
        print('Evaluation session finished.')
        raise NotImplementedError

    def get_src(self) -> str:
        url = f'{self.base_url}/get'
        raise NotImplementedError

    def send_hypo(self, hypo: str) -> None:
        url = f'{self.base_url}/send?hypo={hypo}'
        raise NotImplementedError


def main():
    with SimulSTEvaluationService():
        # read inputs and write predictions
        pass


if __name__ == '__main__':
    main()
