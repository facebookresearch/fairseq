"""
Implement unsupervised metric for decoding hyperparameter selection:
    $$ alpha * LM_PPL + ViterbitUER(%) * 100 $$
"""
import argparse
import logging
import math
import sys

import kenlm
import editdistance
from g2p_en import G2p

logging.root.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_tra", help="reference pseudo labels")
    parser.add_argument("hyp_tra", help="decoded pseudo labels to be assess")
    parser.add_argument("--kenlm_path", default="/checkpoint/abaevski/data/speech/libri/librispeech_lm_novox.phnc_o5.bin", help="")
    parser.add_argument("--uppercase", action="store_true", help="")
    parser.add_argument("--skipwords", default="", help="")
    parser.add_argument("--gt_tra", default="", help="ground truth pseudo labels for computing oracle WER")
    parser.add_argument("--min_vt_uer", default=0.0, type=float)
    parser.add_argument("--phonemize", action="store_true", help="phonemize word hypotheses, used when reference is phone transcript")
    parser.add_argument("--phonemize_lexicon", default="", type=str, help="use a lexicon for phonemizing")
    return parser

def load_tra(tra_path):
    with open(tra_path, "r") as f:
        uid_to_tra = {}
        for line in f:
            toks = line.rstrip().split()
            uid, tra = toks[0], " ".join(toks[1:])
            uid_to_tra[uid] = tra
    logger.debug(f"loaded {len(uid_to_tra)} utterances from {tra_path}")
    return uid_to_tra

def load_lex(lex_path):
    with open(lex_path, "r") as f:
        w2p = {}
        for line in f:
            w, p = line.rstrip().split(None, 1)
            w2p[w] = p.split()
    return w2p
            
def compute_wer(ref_uid_to_tra, hyp_uid_to_tra, g2p, g2p_dict):
    d_cnt = 0
    w_cnt = 0
    w_cnt_h = 0
    for uid in hyp_uid_to_tra:
        ref = ref_uid_to_tra[uid].split()
        if g2p_dict is not None:
            hyp = []
            for word in hyp_uid_to_tra[uid].split():
                if word in g2p_dict:
                    hyp = hyp + g2p_dict[word]
                else:
                    logger.warning(f"{word} not in g2p_dict")
        elif g2p is not None:
            hyp = g2p(hyp_uid_to_tra[uid])
            hyp = [p for p in hyp if p != "'" and p != " "]
            hyp = [p[:-1] if p[-1].isnumeric() else p for p in hyp]
        else:
            hyp = hyp_uid_to_tra[uid].split()
        logger.debug((
            f"======================\n"
            f"HYP: {' '.join(hyp)}\n"
            f"REF: {' '.join(ref)}"
        ))
        d_cnt += editdistance.eval(ref, hyp)
        w_cnt += len(ref)
        w_cnt_h += len(hyp)
    wer = float(d_cnt) / w_cnt
    logger.debug((
        f"wer = {wer*100:.2f}%; num. of ref words = {w_cnt}; "
        f"num. of hyp words = {w_cnt_h}; num. of sentences = {len(ref_uid_to_tra)}"
    ))
    return wer

def compute_lm_ppl(hyp_uid_to_tra, score_fn):
    lm_score = 0.
    w_cnt = 0
    for hyp in hyp_uid_to_tra.values():
        cur_score = score_fn(hyp)
        cur_cnt = len(hyp.split()) + 1  # plus one for </s>
        lm_score += cur_score
        w_cnt += cur_cnt
        logger.debug((
            f"======================\n"
            f"score sum/avg = {cur_score:.2f}/{cur_score/cur_cnt:.2f}\n"
            f"hyp = {hyp}"
        ))
    lm_ppl = math.pow(10, -lm_score / w_cnt)
    logger.debug(f"lm ppl = {lm_ppl:.2f}; num. of words = {w_cnt}")
    return lm_ppl

def main():
    args = get_parser().parse_args()
    logger.debug(f"Args: {args}")
    
    ref_uid_to_tra = load_tra(args.ref_tra)
    hyp_uid_to_tra = load_tra(args.hyp_tra)
    assert not bool(set(hyp_uid_to_tra.keys()) - set(ref_uid_to_tra.keys()))

    lm = kenlm.Model(args.kenlm_path)
    skipwords = set(args.skipwords.split(","))
    def compute_lm_score(s):
        s = " ".join(w for w in s.split() if w not in skipwords)
        s = s.upper() if args.uppercase else s
        return lm.score(s)

    g2p, g2p_dict = None, None
    if args.phonemize:
        if args.phonemize_lexicon:
            g2p_dict = load_lex(args.phonemize_lexicon)
        else:
            g2p = G2p()

    wer = compute_wer(ref_uid_to_tra, hyp_uid_to_tra, g2p, g2p_dict)
    lm_ppl = compute_lm_ppl(hyp_uid_to_tra, compute_lm_score)
    
    gt_wer = -math.inf
    if args.gt_tra:
        gt_uid_to_tra = load_tra(args.gt_tra)
        gt_wer = compute_wer(gt_uid_to_tra, hyp_uid_to_tra, None, None)

    score = math.log(lm_ppl) * max(wer, args.min_vt_uer)
    logging.info(f"{args.hyp_tra}: score={score:.4f}; wer={wer*100:.2f}%; lm_ppl={lm_ppl:.4f}; gt_wer={gt_wer*100:.2f}%")

if __name__ == "__main__":
    main()
