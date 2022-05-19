import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='Source language')
parser.add_argument('--tgt', type=str, help='Target language')
parser.add_argument('--src-file', type=str, help='Input source file')
parser.add_argument('--tgt-file', type=str, help='Input target file')
parser.add_argument('--src-output-file', type=str, help='Output source file')
parser.add_argument('--tgt-output-file', type=str, help='Output target file')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
parser.add_argument('--threshold-character', type=str, default=']', help='Threshold character')
parser.add_argument('--histograms', type=str, help='Path to histograms')

args = parser.parse_args()


def read_hist(f):
    ch = []
    for line in f:
        c = line[0]
        if c == args.threshold_character:
            break
        ch.append(c)
    return ch


with(open("{}/{}".format(args.histograms, args.src), 'r', encoding='utf8')) as f:
    ch1 = read_hist(f)

with(open("{}/{}".format(args.histograms, args.tgt), 'r', encoding='utf8')) as f:
    ch2 = read_hist(f)

print("Accepted characters for {}: {}".format(args.src, ch1))
print("Accepted characters for {}: {}".format(args.tgt, ch2))

with open(args.src_file, 'r', encoding='utf8') as fs1, open(args.tgt_file, 'r', encoding='utf8') as fs2, open(args.src_output_file, 'w', encoding='utf8') as fos1, open(args.tgt_output_file, 'w', encoding='utf8') as fos2:
    ls1 = fs1.readline()
    ls2 = fs2.readline()

    while ls1 or ls2:
        cnt1 = len([c for c in ls1.strip() if c in ch1])
        cnt2 = len([c for c in ls2.strip() if c in ch2])

        if cnt1 / len(ls1) > args.threshold and cnt2 / len(ls2) > args.threshold:
            fos1.write(ls1)
            fos2.write(ls2)
        else:
            print("{} {} {} \n{} {} {}".format(args.src, cnt1 / len(ls1), ls1.strip(), args.tgt, cnt2 / len(ls2), ls2.strip()))

        ls1 = fs1.readline()
        ls2 = fs2.readline()
        