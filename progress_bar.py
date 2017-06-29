"""
Progress bar wrapper around tqdm which handles non-tty outputs
"""
import sys
from tqdm import tqdm


class progress_bar(tqdm):
    enabled = sys.stderr.isatty()

    def __new__(cls, *args, **kwargs):
        if cls.enabled:
            return tqdm(*args, **kwargs)
        else:
            return simple_progress_bar(*args, **kwargs)


class simple_progress_bar(tqdm):
    print_interval = 1000

    def __init__(self, *args, **kwargs):
        super(simple_progress_bar, self).__init__(*args, **kwargs)

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if i > 0 and i % self.print_interval == 0:
                msg = '{} {:5d} / {:d} {}\n'.format(self.desc, i, size, self.postfix)
                sys.stdout.write(msg)

    @classmethod
    def write(cls, s, file=None, end="\n"):
        fp = file if file is not None else sys.stdout
        fp.write(s)
        fp.write(end)

    @staticmethod
    def status_printer(file):
        def print_status(s):
            pass
        return print_status
