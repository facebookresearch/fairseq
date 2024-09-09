import argparse
import json
from collections import defaultdict
import os
from tqdm import tqdm
import editdistance
import werpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--hyp', type=str)  # asr
    parser.add_argument('--ref', type=str)  # asr
    parser.add_argument('--ref_lid', type=str)  # ref lid
    parser.add_argument('--hyp_lid', type=str)  # hyp lid
    parser.add_argument('--topk_lid', type=str)  # topk lid
    parser.add_argument('--k', type=str, default=10)
    parser.add_argument('--lc', type=int, default=0)  # lowercase
    parser.add_argument('--rm', type=int, default=0)  # remove punc
    parser.add_argument('--wer', type=int, default=0)  # WER
    parser.add_argument('--exclude', nargs="*", default=None)  # exclude langs
    args = parser.parse_args()

    hyps = [x.strip() for x in open(args.hyp, "r").readlines()]
    refs = [x.strip() for x in open(args.ref, "r").readlines()]
    assert len(hyps) == (len(refs) * args.k)

    langs = [x.strip() for x in open(args.ref_lid, "r").readlines()]
    assert len(langs) == len(refs)

    confusions = [x.strip() for x in open(args.hyp_lid, "r").readlines()]
    assert len(confusions) == len(refs)

    topk_langs = [x.strip() for x in open(args.topk_lid, "r").readlines()]
    assert len(topk_langs) == len(hyps)

    if args.wer != 0:
        cer_langs = [x.strip() for x in open("/private/home/yanb/MMS1_public/fairseq/examples/mms/asr/data/cer_langs.txt", "r").readlines()]

    total_errors = defaultdict(int)
    total_length = defaultdict(int)

    subset_total_errors = defaultdict(int)
    subset_total_length = defaultdict(int)

    lid_correct = defaultdict(int)
    lid_total = defaultdict(int)

    subset_lid_correct = defaultdict(int)
    subset_lid_total = defaultdict(int)

    error_rates = []
    errors = []
    lens = []

    num_utts = 0
    empty = 0
    for i, hyp in tqdm(enumerate(hyps)):
        if args.exclude is not None:
            if langs[i // args.k] in args.exclude:
                error_rates.append("exclude")
                errors.append("exclude")
                lens.append("exclude")
                continue

        ref = refs[i // args.k]
        which_k = i % args.k

        if args.lc != 0:
            hyp = hyp.lower()
            ref = ref.lower()
        
        if args.rm != 0:
            hyp = hyp.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")
            ref = ref.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "")

        if args.wer != 0 and langs[i // args.k] in cer_langs:
            hyp = " ".join(hyp)
            ref = " ".join(ref)

        hyp_words = hyp.split()
        tgt_words = ref.split()
        
        # skip empty refs
        if ref == "":
            empty += 1
            continue
        if ref.strip() == "":
            empty += 1
            continue

        # check sub/ins/del with another pkg
        # try:
        # summary = werpy.summary(ref, hyp)
        # except:
            # import pdb;pdb.set_trace()
        # errs = summary.ld[0]
        errs = editdistance.eval(hyp_words, tgt_words)

        # ASR
        total_errors[which_k] += errs
        total_length[which_k] += len(tgt_words)

        if langs[i // args.k] != confusions[i // args.k]:
            num_utts += 1
            subset_total_errors[which_k] += errs
            subset_total_length[which_k] += len(tgt_words)

        error_rates.append(errs / len(tgt_words))
        errors.append(errs)
        lens.append(len(tgt_words))

        # LID
        if topk_langs[i] == langs[i // args.k]:
            lid_correct[which_k] += 1
        lid_total[which_k] += 1

        if langs[i // args.k] != confusions[i // args.k]:
            if topk_langs[i] == langs[i // args.k]:
                subset_lid_correct[which_k] += 1
            subset_lid_total[which_k] += 1

    print(empty)
    print(num_utts)

    if args.exclude is not None:
        if len(args.exclude) > 50:
            exclude_tag = ".exclude_" + str(len(args.exclude))
        else:
            exclude_tag = ".exclude_" + "_".join(args.exclude)
    else:
        exclude_tag = ""

    if args.lc == 1:
        lc_tag = ".lc"
    else:
        lc_tag = ""
    
    if args.rm == 1:
        rm_tag = ".rm"
    else:
        rm_tag = ""

    # import pdb;pdb.set_trace()
    with open(args.hyp + ".result" + exclude_tag + lc_tag + rm_tag, "w") as f:
        f.write("AGGREGATE ASR ERROR RATE\n")
        for k in total_length:
            f.write(str(k) + "\t" + '{:.4g}'.format(total_errors[k] * 100 / total_length[k]) + "\n")

        f.write("\nLID-ERR SUBSET ASR ERROR RATE\n")
        for k in total_length:
            f.write(str(k) + "\t" + '{:.4g}'.format(subset_total_errors[k] * 100 / subset_total_length[k]) + "\n")

        f.write("\nAGGREGATE LID ACC\n")
        for k in total_length:
            f.write(str(k) + "\t" + '{:.4g}'.format(lid_correct[k] * 100 / lid_total[k]) + "\n")

        f.write("\nLID-ERR SUBSET LID ACC\n")
        for k in total_length:
            f.write(str(k) + "\t" + '{:.4g}'.format(subset_lid_correct[k] * 100 / subset_lid_total[k]) + "\n")

        # compute min-wer-oracle candidate selection
        
        assert len(error_rates) % args.k == 0
        ks = [1,2,3,4,5,6,7,8,9,10]
        oracle_res = defaultdict(float)
        subset_oracle_res = defaultdict(float)
        for k in ks:
            oracle_errs = 0
            oracle_len = 0
            subset_oracle_errs = 0
            subset_oracle_len = 0
            tmp = 0
            for i in range(len(error_rates) // args.k):
                if args.exclude is not None:
                    if langs[i] in args.exclude:
                        continue

                j = i * args.k
                cands = error_rates[j:j+k]
                try:
                    min_idx, min_val = min(enumerate(cands), key=lambda x: x[1])
                except:
                    import pdb;pdb.set_trace()
                oracle_errs += errors[j + min_idx]
                oracle_len += lens[j + min_idx]

                if langs[i] != confusions[i]:
                    tmp += 1
                    subset_oracle_errs += errors[j + min_idx]
                    subset_oracle_len += lens[j + min_idx]

            print(tmp)
            oracle_res[k] = oracle_errs * 100 / oracle_len
            subset_oracle_res[k] = subset_oracle_errs * 100 / subset_oracle_len
                
        f.write("\nMIN-WER-ORACLE ASR ERROR RATE\n")
        for k in oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(oracle_res[k]) + '\n')

        f.write("\nLID-ERR SUBSET MIN-WER-ORACLE ASR ERROR RATE\n")
        for k in subset_oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(subset_oracle_res[k]) + '\n')

        
        # compute lid-oracle candidate selection

        oracle_res = defaultdict(float)
        subset_oracle_res = defaultdict(float)
        oracle_not_in_k = defaultdict(float)
        subset_oracle_not_in_k = defaultdict(float)
        # ks = [10]
        for k in ks:
            oracle_errs = 0
            oracle_len = 0
            subset_oracle_errs = 0
            subset_oracle_len = 0
            tmp = 0
            not_in_k = 0
            subset_not_in_k = 0
            cnt = 0
            subset_cnt = 0
            for i in range(len(error_rates) // args.k):
                if args.exclude is not None:
                    if langs[i] in args.exclude:
                        continue

                cnt += 1
                j = i * args.k
                cands = error_rates[j:j+k]
                cands_lid = topk_langs[j:j+k]
                try:
                    min_idx = cands_lid.index(langs[i])
                except:
                    # import pdb;pdb.set_trace()
                    min_idx = 0
                    not_in_k += 1
                    if langs[i] != confusions[i]:
                        subset_not_in_k += 1
                oracle_errs += errors[j + min_idx]
                oracle_len += lens[j + min_idx]

                if langs[i] != confusions[i]:
                    tmp += 1
                    subset_cnt += 1
                    subset_oracle_errs += errors[j + min_idx]
                    subset_oracle_len += lens[j + min_idx]
            
            print(tmp)
            oracle_res[k] = oracle_errs * 100 / oracle_len
            subset_oracle_res[k] = subset_oracle_errs * 100 / subset_oracle_len
            oracle_not_in_k[k] = not_in_k * 100 / cnt
            subset_oracle_not_in_k[k] = subset_not_in_k * 100 / subset_cnt

        f.write("\nLID-ORACLE ASR ERROR RATE\n")
        for k in oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(oracle_res[k]) + '\n')

        f.write("\nLID-ORACLE NOT-IN-K RATE\n")
        for k in oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(oracle_not_in_k[k]) + '\n')

        f.write("\nLID-ERR SUBSET LID-ORACLE ASR ERROR RATE\n")
        for k in subset_oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(subset_oracle_res[k]) + '\n')

        f.write("\nLID-ERR SUBSET NOT-IN-K RATE\n")
        for k in subset_oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(subset_oracle_not_in_k[k]) + '\n')

        
        # compute non-blank cand sel

        ks = [1,2,3,4,5,6,7,8,9,10]
        oracle_res = defaultdict(float)
        subset_oracle_res = defaultdict(float)

        for k in ks:
            oracle_errs = 0
            oracle_len = 0
            subset_oracle_errs = 0
            subset_oracle_len = 0
            tmp = 0
            for i in range(len(error_rates) // args.k):
                if args.exclude is not None:
                    if langs[i] in args.exclude:
                        continue

                j = i * args.k
                cands = hyps[j:j+k]
                try:
                    min_idx = 0
                    for idx, c in enumerate(cands):
                        if cands[min_idx] == "" and c != "":
                            min_idx = idx
                            break
                except:
                    import pdb;pdb.set_trace()
                oracle_errs += errors[j + min_idx]
                oracle_len += lens[j + min_idx]

                if langs[i] != confusions[i]:
                    tmp += 1
                    subset_oracle_errs += errors[j + min_idx]
                    subset_oracle_len += lens[j + min_idx]

            print(tmp)
            oracle_res[k] = oracle_errs * 100 / oracle_len
            subset_oracle_res[k] = subset_oracle_errs * 100 / subset_oracle_len
                
        f.write("\nNON-BLANK ASR ERROR RATE\n")
        for k in oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(oracle_res[k]) + '\n')

        f.write("\nLID-ERR SUBSET NON-BLANK ASR ERROR RATE\n")
        for k in subset_oracle_res:
            f.write(str(k) + '\t' + '{:.4g}'.format(subset_oracle_res[k]) + '\n')