#!/usr/bin/env python3

import sys

from sacremoses.normalize import MosesPunctNormalizer


def main(args):
    normalizer = MosesPunctNormalizer(lang=args.lang, penn=args.penn)
    for line in sys.stdin:
        print(normalizer.normalize(line.rstrip()), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', default='en')
    parser.add_argument('--penn', '-p', action='store_true')
    args = parser.parse_args()

    main(args)
