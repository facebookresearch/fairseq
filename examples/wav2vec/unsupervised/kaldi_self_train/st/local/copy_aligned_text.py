import sys

for idx, line in enumerate(sys.stdin):
    print(f"utt{idx:010d} {line}", end='')