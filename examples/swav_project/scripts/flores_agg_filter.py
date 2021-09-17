
import argparse
import os
import numpy as np
import itertools
import regex

# U+2581
spm_underscore = "▁"
spm_underscore2 = u"\u2581"
# unicode block
# \p{InDevanagari}: U+0900–U+097F
# x = r'\p{InDevanagari}'
# regex.match(r'\p{InDevanagari}+', x)

def devanagariratio_filter(lines, pct, vocab, vocab_tup):
    pct = float(pct)
    pct_r = pct / 100
    new_lines = []
    discarded = []
    for l in lines:
        toks = [x.replace(spm_underscore, "") for x in l.split(" ")]
        deva_match = [x for x in toks if regex.match(r'\p{InDevanagari}+', x) is not None]
        deva_rate = len(deva_match) / float(len(toks))
        if deva_rate > pct_r:
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'devanagari_ratio_filter {pct}: filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded


def sinhalaratio_filter(lines, pct, vocab, vocab_tup):
    pct = float(pct)
    pct_r = pct / 100
    new_lines = []
    discarded = []
    for l in lines:
        toks = [x.replace(spm_underscore, "") for x in l.split(" ")]
        deva_match = [x for x in toks if regex.match(r'\p{InSinhala}+', x) is not None]
        deva_rate = len(deva_match) / float(len(toks))
        if deva_rate > pct_r:
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'sinhalaratio_filter {pct}: filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded


def alpharatio_filter(lines, pct, vocab, vocab_tup):
    pct = float(pct)
    pct_r = pct / 100
    new_lines = []
    discarded = []
    for l in lines:
        toks = [x.replace(spm_underscore, "") for x in l.split(" ")]
        alpha_match = [x for x in toks if regex.match(r'[a-zA-Z]+', x) is not None]
        alpha_rate = len(alpha_match) / float(len(toks))
        if alpha_rate > pct_r:
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'alpharatio {pct}: filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded


def vocabtop_filter(lines, pct, vocab, vocab_tup):
    pct = float(pct)
    chunk_vocab_tup = vocab_tup[:int(len(vocab_tup) * pct / 100)]
    chunk_words = set(x[0] for x in chunk_vocab_tup)
    new_lines = []
    discarded = []
    for l in lines:
        toks = l.split(" ")
        if all(t in chunk_words for t in toks):
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'vocabtop_filter {pct}: filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded


def vocabfreq_filter(lines, freq, vocab, vocab_tup):
    freq = int(freq)
    chunk_vocab_tup = [x for x in vocab_tup if x[1] > freq]
    chunk_words = set(x[0] for x in chunk_vocab_tup)
    new_lines = []
    discarded = []
    for l in lines:
        toks = l.split(" ")
        if all(t in chunk_words for t in toks):
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'vocabfreq_filter {freq}: filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded


def rm_alpha(lines, params, vocab, vocab_tup):
    new_lines = []
    discarded = []
    for l in lines:
        toks = l.split(" ")
        toks = [x[1:] if x.startswith(spm_underscore) else x for x in toks]
        if not any(x.isalpha() for x in toks):
            new_lines.append(l)
        else:
            discarded.append(l)
    print(f'rm_alpha : filtered {len(lines)} -> {len(new_lines)} (retained {len(new_lines) / float(len(lines))})')
    return new_lines, discarded



def filter_abstract(lines, ftype, vocab, vocab_tup):
    fname, params = ftype.split("_")
    if fname == "vocabtop":
        return vocabtop_filter(lines, params, vocab, vocab_tup)
    elif fname == "vocabfreq":
        return vocabfreq_filter(lines, params, vocab, vocab_tup)
    elif fname == "rm" and params == "alpha":
        return rm_alpha(lines, params, vocab, vocab_tup)
    elif fname == "devanagariratio":
        return devanagariratio_filter(lines, params, vocab, vocab_tup)
    
    elif fname == "sinhalaratio":
        return sinhalaratio_filter(lines, params, vocab, vocab_tup)
    
    elif fname == "alpharatio":
        return alpharatio_filter(lines, params, vocab, vocab_tup)
    else:
        raise ValueError(f'{fname} wrong.')


def filter_process(input_file, output_file, filter_type, discard):
    fil_types = filter_type.split(",")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.read().splitlines()]
    
    has_vocab_filter = any("vocab" in x for x in fil_types)
    if has_vocab_filter:
        print(f'Building vocab from {len(lines)} lines')
        toks = list(itertools.chain.from_iterable(x.strip().split(" ") for x in lines))
        vocab = {}
        for t in toks:
            vocab[t] = vocab.get(t, 0) + 1
        print(f'Vocab size: {len(vocab)}')
        vocab_tup = [(k, v) for k, v in vocab.items()]
        vocab_tup = sorted(vocab_tup, key=lambda x: x[1], reverse=True)
        print(f'Top  10 words: {vocab_tup[:10]}')
        print(f'Last 10 words: {vocab_tup[-10:]}')
    else:
        print(f'Skip building vocab because no filter require it')
        vocab = vocab_tup = None

    ori_size = len(lines)
    discard_lines = []
    for fidx, ftype in enumerate(fil_types):
        print(f'--Start filter {fidx} - {ftype}......')
        lines, discarded = filter_abstract(lines, ftype, vocab, vocab_tup)
        discard_lines.extend(discarded)
    end_size = len(lines)
    print(f'All filtered {ori_size} -> {end_size} ({end_size / float(ori_size)})')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    with open(discard, 'w', encoding='utf-8') as f:
        f.write('\n'.join(discard_lines))
    
    print(f'Finish saving to {output_file}')
    print(f'Finish saving to {discard}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--discard', type=str)
    parser.add_argument('--filter', type=str, 
        default="vocabtop_80,vocabfreq_2,rm_alpha",
    )
    

    args = parser.parse_args()
    filter_process(args.input, args.output, args.filter, args.discard)








