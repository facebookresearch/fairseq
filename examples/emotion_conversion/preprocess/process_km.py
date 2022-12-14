import sys
import argparse
from tqdm import tqdm
from build_emov_translation_manifests import dedup, remove_under_k


if __name__ == "__main__":
    """
    this is a standalone script to process a km file
    specifically, to dedup or remove tokens that repeat less
    than k times in a row
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("km", type=str, help="path to km file")
    parser.add_argument("--dedup", action='store_true')
    parser.add_argument("--remove-under-k", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not args.dedup and args.remove_under_k == 0:
        print("nothing to do! quitting...")
        sys.exit(0)

    km = open(args.km, "r").readlines()
    out = []
    for line in tqdm(km):
        if args.remove_under_k > 0:
            line = remove_under_k(line, args.remove_under_k)
        if args.dedup:
            line = dedup(line)
        out.append(line)

    path = args.km if args.output is None else args.output
    if args.remove_under_k > 0:
        path = path.replace(".km", f"-k{args.remove_under_k}.km")
    if args.dedup:
        path = path.replace(".km", f"-deduped.km")

    open(path, "w").writelines(out)
    print(f"written to {path}")
