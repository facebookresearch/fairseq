import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--coverage', type=str)
    parser.add_argument('--correct', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--mode', type=int, default=0)  # 0=incorrect+cov 1=correct+cov
    args = parser.parse_args()

    coverage = [x.strip() for x in open(args.coverage, "r").readlines()]
    correct = [x.strip() for x in open(args.correct, "r").readlines()]

    merged = []
    for i in range(len(coverage)):
        # incorrect and covered --> 1
        # incorrect and not covered --> 0
        if args.mode == 0:
            merged.append(int(coverage[i] == "1" and correct[i] == "0"))

        # correct and covered --> 1
        # correct and not covered --> 0
        elif args.mode == 1:
            merged.append(int(coverage[i] == "1" and correct[i] == "1"))
        else:
            print("Mode not supported")
            break

    with open(args.dst, "w") as f:
        f.writelines([str(x)+"\n" for x in merged])