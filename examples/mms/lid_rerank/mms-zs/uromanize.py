import os
import tempfile
import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--txt", type=str)
parser.add_argument("--lid", type=str)
parser.add_argument("--dst", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()

UROMAN_PL = args.model + "uroman/bin/uroman.pl"

def norm_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()

def uromanize(words):
    iso = "xxx"
    with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
        with open(tf.name, "w") as f:
            f.write("\n".join(words))
        cmd = f"perl " + UROMAN_PL
        cmd += f" -l {iso} "
        cmd += f" < {tf.name} > {tf2.name}"
        os.system(cmd)
        lexicon = {}
        with open(tf2.name) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                line = re.sub(r"\s+", "", norm_uroman(line)).strip()
                lexicon[words[idx]] = " ".join(line) + " |"
    return lexicon

def convert_sent(txt, char_lang=False):
    if char_lang:
        words = txt
    else:
        words = txt.split(" ")
    lexicon = uromanize(words)
    pron = []
    pron_no_sp = []
    for w in words:
        if w in lexicon:
            pron.append(lexicon[w])
            pron_no_sp.append(lexicon[w].replace(" |", ""))

    return " ".join(pron), " ".join(pron_no_sp)

if __name__ == "__main__":
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    txts = [x.strip() for x in open(args.txt, "r").readlines()]
    langs = [x.strip() for x in open(args.lid, "r").readlines()]
    assert len(txts) == len(langs)

    cer_langs = [x.strip() for x in open("cer_langs.txt", "r").readlines()]

    with open(args.dst + "/nbest_asr_hyp_uroman", "w", buffering=1) as f:
        for t, l in tqdm(zip(txts,langs), total=len(txts)):
            pron, _ = convert_sent(t, l in cer_langs)
            f.write(pron + "\n")
