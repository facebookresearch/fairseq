import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--mapping', type=str)
    args = parser.parse_args()

    predictions = [eval(x.strip()) for x in open(args.predictions, "r").readlines()]
    # whisper_lid_code:mms_lid_code
    mapping = {x[0]:x[1] for x in [l.strip().split(":", 1) for l in open(args.mapping, "r").readlines()]}
    predictions2 = []
    for p in predictions:
        new_p = []
        for x in p:
            new_p.append([mapping[x[0]], x[1]])
        predictions2.append(new_p)

    with open(args.predictions + "2", "w") as f:
        f.writelines([str(x) + "\n" for x in predictions2])