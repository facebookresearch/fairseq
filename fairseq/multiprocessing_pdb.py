def multiprocessing_pdb():
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """
    import pdb
    import sys
    class MultiprocessingPdb(pdb.Pdb):
        def interaction(self, *args, **kwargs):
            orig_stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = orig_stdin
    return MultiprocessingPdb()


pdb = multiprocessing_pdb()
