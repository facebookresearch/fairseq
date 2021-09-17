import os
import subprocess
import logging
import argparse

logger = logging.getLogger(__name__)
BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--encoding', type=str, default='utf-8')

    args = parser.parse_args()

    bleu = eval_moses_bleu(args.ref, args.hyp)
    print(f'BLEU: {bleu}')
