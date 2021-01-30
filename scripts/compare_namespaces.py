#!/usr/bin/env python
"""Helper script to compare two argparse.Namespace objects."""

from argparse import Namespace  # noqa


def main():

    ns1 = eval(input("Namespace 1: "))
    ns2 = eval(input("Namespace 2: "))

    def keys(ns):
        ks = set()
        for k in dir(ns):
            if not k.startswith("_"):
                ks.add(k)
        return ks

    k1 = keys(ns1)
    k2 = keys(ns2)

    def print_keys(ks, ns1, ns2=None):
        for k in ks:
            if ns2 is None:
                print("{}\t{}".format(k, getattr(ns1, k, None)))
            else:
                print(
                    "{}\t{}\t{}".format(k, getattr(ns1, k, None), getattr(ns2, k, None))
                )

    print("Keys unique to namespace 1:")
    print_keys(k1 - k2, ns1)
    print()

    print("Keys unique to namespace 2:")
    print_keys(k2 - k1, ns2)
    print()

    print("Overlapping keys with different values:")
    ks = [k for k in k1 & k2 if getattr(ns1, k, "None") != getattr(ns2, k, "None")]
    print_keys(ks, ns1, ns2)
    print()


if __name__ == "__main__":
    main()
