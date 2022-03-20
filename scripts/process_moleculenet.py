from sklearn.model_selection import train_test_split

import pandas as pd
import argparse
import os

EMPTY_INDEX = -1

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

p.add_argument("--input", help="input file", type=str)
p.add_argument("--smile-index", type=int, required=True)
p.add_argument("--class-index", type=int, required=True)

p.add_argument("--delimiter", type=str, default=",")

p.add_argument("--train-perc", type=float, default=0.8)
p.add_argument("--valid-perc", type=float, default=0.1)
p.add_argument("--test-perc", type=float, default=0.1)

p.add_argument("--seed", type=int, default=42)
p.add_argument("--filter", type=bool, default=False)

args = p.parse_args()

data_frame = pd.read_csv(args.input, delimiter=args.delimiter)

if args.filter:
    nan_value = float("NaN")
    data_frame.replace("", nan_value, inplace=True)
    data_frame.dropna(subset=[data_frame.columns[args.class_index], data_frame.columns[args.smile_index]], inplace=True)
else:
    data_frame.replace("", EMPTY_INDEX, inplace=True)
    data_frame.fillna(EMPTY_INDEX, inplace=True)


print(data_frame)

smiles_data = list(map(str, data_frame.iloc[:, args.smile_index].tolist()))
class_data = list(map(int, data_frame.iloc[:, args.class_index].tolist()))

X_train, X_valid, y_train, y_valid = train_test_split(
    smiles_data, class_data, test_size=args.valid_perc + args.test_perc, random_state=args.seed)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_valid, y_valid, test_size=args.test_perc/(args.valid_perc + args.test_perc), random_state=args.seed)

print(f"Train Length: {len(X_train)}")
print(f"Valid Length: {len(X_valid)}")
print(f"Test Length: {len(X_test)}")

names = ["train", "valid", "test"]
X_splits = []
y_splits = []


os.system("mkdir raw")
os.system("mkdir tokenized")
os.system("mkdir processed")

# Write Raw Splits
print("Writing Input Splits")
for name, smiles in zip(names, (X_train, X_valid, X_test)):
    new_path = "raw/" + args.input + "." + name + ".input"
    X_splits.append(new_path)
    with open(new_path, "w+") as f:
        for smile in smiles:
            f.write(f"{smile}\n")

print("Writing Output Splits")
for name, targets in zip(names, (y_train, y_valid, y_test)):
    new_path = "raw/" + args.input + "." + name + ".target"
    y_splits.append(new_path)
    with open(new_path, "w+") as f:
        for target in targets:
            f.write(f"{str(target)}\n")

# Tokenize Texts
print("Tokenizing")
splits = []
for path in X_splits:
    cur_path = path.replace('raw', 'tokenized')
    splits.append(cur_path)
    os.system(
        f"python /data/home/armenag/code/chem/fairseq-py/scripts/spm_parallel.py --input {path} --outputs {cur_path} --model /fsx-html2/armenag/chemical/tokenizer/chem.model")

X_splits = splits

os.system(('fairseq-preprocess --only-source '
           f'--trainpref "{X_splits[0]}" '
           f'--validpref "{X_splits[1]}" '
           f'--testpref "{X_splits[2]}" '
           '--destdir "processed/input0" --workers 60 '
           '--srcdict /fsx-html2/armenag/chemical/tokenizer/chem.vocab.fs'))
os.system(('fairseq-preprocess '
           '--only-source '
           f'--trainpref "{y_splits[0]}" '
           f'--validpref "{y_splits[1]}" '
           f'--testpref "{y_splits[2]}" '
           f'--destdir "processed/label" '
           '--workers 60;)'))
