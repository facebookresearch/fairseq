import argparse
import json
from collections import defaultdict
import os
import soundfile as sf
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--wavs_tsv', type=str)
    parser.add_argument('--lid_preds', type=str)
    parser.add_argument('--txt', type=str)
    parser.add_argument('--refs', type=str, default=None)
    parser.add_argument('--langs', type=str, default=None)
    parser.add_argument('--confs', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    args = parser.parse_args()

    # split wavs into dst/lang/wav.txt and dst/lang/ids.txt
    # uses lid_preds to create topk asr; 1 wav has k different lid

    wavs_tsv = [x for x in open(args.wavs_tsv, "r").readlines()]
    root = wavs_tsv[0]
    wavs = wavs_tsv[1:]
    lid_preds = [eval(x) for x in open(args.lid_preds, "r").readlines()]
    txts = [x.strip() for x in open(args.txt, "r").readlines()]
    if args.refs is not None:
        refs = [x.strip() for x in open(args.refs, "r").readlines()]
        assert len(wavs) == len(refs)
        refs_filt = []
    if args.langs is not None:
        langs = [x.strip() for x in open(args.langs, "r").readlines()]
        assert len(wavs) == len(langs)
        langs_filt = []
    if args.confs is not None:
        confs = [x.strip() for x in open(args.confs, "r").readlines()]
        assert len(wavs) == len(confs)
        confs_filt = []

    assert len(wavs) == len(lid_preds)
    

    topk_wavs = []
    topk_langs = []
    topk_txts = []

    
    for i, (w, p) in enumerate(zip(wavs, lid_preds)):
        if p == "n/a":
            print("!")
            continue
        
        assert len(p) == len(lid_preds[0])
        txt_start_idx = i * len(lid_preds[0])
        txt_end_idx = txt_start_idx + len(lid_preds[0])
        cand_txts = txts[txt_start_idx:txt_end_idx]
        assert len(cand_txts) == len(lid_preds[0])

        for (l, _), t in zip(p, cand_txts):
            topk_wavs.append(w)
            topk_langs.append(l)
            topk_txts.append(t)

        if args.refs is not None:
            refs_filt.append(refs[i])
        if args.langs is not None:
            langs_filt.append(langs[i])
        if args.confs is not None:
            confs_filt.append(confs[i])

    lang_split = defaultdict(list)
    for id, (wav,lid,txt) in enumerate(zip(topk_wavs, topk_langs, topk_txts)):
        lang_split[lid].append((id, wav, txt))

    for lang in tqdm(lang_split.keys()):
        if not os.path.exists(args.dst + "/" + lang):
            os.makedirs(args.dst + "/" + lang)

        with open(args.dst + "/" + lang + "/test.tsv", "w") as f1, \
            open(args.dst + "/" + lang + "/ids.txt", "w") as f2, \
            open(args.dst + "/" + lang + "/txts.txt", "w") as f3:
            f1.write(root)
            f1.writelines([x[1] for x in lang_split[lang]])
            f2.writelines([str(x[0]) + "\n" for x in lang_split[lang]])
            f3.writelines([x[2] + "\n" for x in lang_split[lang]])

        # lc.rm
        tgt = [x[2].lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(")", "").replace("(", "").replace("-", "") for x in lang_split[lang]]
        with open(args.dst + "/" + lang + "/test.ltr", "w") as fw:
            fw.writelines([" ".join(x.replace(" ","|")) + "\n" for x in tgt])
        with open(args.dst + "/" + lang + "/test.wrd", "w") as fw:
            fw.writelines([x + "\n" for x in tgt])

    with open(args.dst + "/lid.txt", "w") as f:
        f.writelines([x+"\n" for x in topk_langs])

    if args.refs is not None:
        with open(args.dst + "/refs.txt", "w") as f:
            f.writelines([x+"\n" for x in refs_filt])
    if args.langs is not None:
        with open(args.dst + "/langs.txt", "w") as f:
            f.writelines([x+"\n" for x in langs_filt])
    if args.confs is not None:
        with open(args.dst + "/confs.txt", "w") as f:
            f.writelines([x+"\n" for x in confs_filt])