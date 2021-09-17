
import glob
from itertools import filterfalse
import os
import sys
import faiss
import argparse
import tempfile
import numpy as np
import torch




###############################################################################
#
# Load texts and remove duplicates
#
###############################################################################

def TextLoadUnify(fname, args):
    if args.verbose:
        print(' - loading texts {:s}: '.format(fname), end='')
    fin = open(fname, encoding=args.encoding, errors='surrogateescape')
    inds = []
    sents = []
    sent2ind = {}
    n = 0
    nu = 0
    for line in fin:
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if args.unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
    if args.verbose:
        print('{:d} lines, {:d} unique'.format(n, nu))
    del sent2ind
    return inds, sents


###############################################################################
#
# Wrapper for knn on CPU/GPU
#
###############################################################################

def knn(x, y, k, use_gpu, mem=5*1024*1024*1024):
    return knnGPU(x, y, k, mem=mem) if use_gpu else knnCPU(x, y, k)


###############################################################################
#
# Perform knn on GPU
#
###############################################################################

def knnGPU(x, y, k, mem=5*1024*1024*1024):
    dim = x.shape[1]
    batch_size = int(mem) // (dim*4)
    sim = np.zeros((x.shape[0], k), dtype=np.float32)
    ind = np.zeros((x.shape[0], k), dtype=np.int64)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            # print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y[yfrom:yto])
            bsim, bind = idx.search(x[xfrom:xto], min(k, yto-yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind


###############################################################################
#
# Perform knn on CPU
#
###############################################################################

def knnCPU(x, y, k):
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(y)
    sim, ind = idx.search(x, k)
    return sim, ind


###############################################################################
#
# Scoring
#
###############################################################################

def score(x, y, fwd_mean, bwd_mean, margin):
    # margin = lambda a, b: a / b
    # x.dot(y) / (mean_fwd_bwd)
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(' - scoring {:d} candidates'.format(x.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def load_pth(name, pth_path, em_key='prototypes'):
    paths = list(glob.glob(pth_path))
    embeds = ids = None
    em_key = "prototypes"
    print(f'Load {name}[{len(paths)}]: {paths}')
    for i, p in enumerate(paths):
        print(f'Loading data {name} {i}: {p}')
        data = torch.load(p, map_location='cpu')
        if em_key not in data:
            em_key = 'embed'
            print(f'prototypes not in data, try using embed')
        _ems = data[em_key].softmax(-1).numpy()
        _ids = data['id'].numpy()
        if embeds is None:
            embeds = _ems
            ids = _ids
        else:
            embeds = np.concatenate((embeds, _ems), 0)
            ids = np.concatenate((ids, _ids), 0)
    return ids, embeds


def build_filter_fn(args):
    len_ratio = args.len_ratio

    def filter_fn(score, src_idx, tgt_idx, align_src_idxs, align_tgt_idxs, align_scores, src_texts, tgt_texts, fsrc=None, ftgt=None):
        assert src_texts is not None
        assert tgt_texts is not None
        accept = True
        if len_ratio > 1:
            # (len_mat > self.onspot_len_ratio) | (len_mat < 1.0 / self.onspot_len_ratio)
            _ratio = float(len(src_texts[src_idx])) / float(len(tgt_texts[tgt_idx]))
            accept &= 1.0 / len_ratio < _ratio < len_ratio
        if accept:
            align_src_idxs.append(src_idx)
            align_tgt_idxs.append(tgt_idx)
            align_scores.append(score)
            if fsrc is not None and ftgt is not None:
                fsrc.write(f'{src_texts[src_idx]}\n')
                ftgt.write(f'{tgt_texts[tgt_idx]}\n')
        return int(accept)
    return filter_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LASER: Mine bitext')
    # parser.add_argument('--src', help='Source language corpus')
    # parser.add_argument('--tgt', help='Target language corpus')
    # parser.add_argument('-d', '--data', help='path to data directory, placeholder, not required')
    parser.add_argument('--encoding', default='utf-8',help='Character encoding for input/output')
    parser.add_argument('--src-lang', required=True, help='Source language id')
    parser.add_argument('--tgt-lang', required=True, help='Target language id')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--src_txt', type=str, default=None)
    parser.add_argument('--tgt_txt', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0, help='Threshold on extracted bitexts')

    # mining params
    parser.add_argument('--mode',
        choices=['search', 'score', 'mine'], required=True,
        help='Execution mode')
    parser.add_argument('-k', '--neighborhood',
        type=int, default=4,
        help='Neighborhood size')
    parser.add_argument('--margin',
        choices=['absolute', 'distance', 'ratio'], default='ratio',
        help='Margin function')
    parser.add_argument('--retrieval',
        choices=['fwd', 'bwd', 'max', 'intersect'], default='max',
        help='Retrieval strategy')
    parser.add_argument('--unify', action='store_true',
        help='Unify texts')
    parser.add_argument('--gpu', action='store_true',
        help='Run knn on all available GPUs')
    parser.add_argument('--verbose', action='store_true',
        help='Detailed output')

    # pth data
    parser.add_argument('--src_pth', required=True,
        help='Precomputed source sentence embeddings')
    parser.add_argument('--tgt_pth', required=True,
        help='Precomputed target sentence embeddings')
    parser.add_argument('--dim', type=int, default=1024,
        help='Embedding dimensionality')
    
    # filtering
    parser.add_argument('--len_ratio', type=float, default=0,
        help='Filter accept 1.0 / len_ratio < lenx/leny < len_ratio')
    
    # mem
    parser.add_argument('--mem', type=int, default=5,
        help='mem, in terms of GB')
        
    parser.add_argument('--em_key', type=str, default="prototypes",
        help='em_key prototypes')

    

    args = parser.parse_args()

    if args.gpu:
        print(' - knn will run on all available GPUs (recommended)')
    else:
        print(' - knn will run on CPU (slow)')
    
    if args.src_txt is not None and args.tgt_txt is not None:
        assert os.path.exists(args.src_txt), f'{args.src_txt} not found.'
        assert os.path.exists(args.tgt_txt), f'{args.tgt_txt} not found.'

    src_ids, src_embeds = load_pth(f"src-{args.src_lang}", args.src_pth)
    tgt_ids, tgt_embeds = load_pth(f"tgt-{args.tgt_lang}", args.tgt_pth)

    faiss.normalize_L2(src_embeds)
    faiss.normalize_L2(tgt_embeds)
    
    # calculate knn in both directions
    x = src_embeds
    y = tgt_embeds
    if args.retrieval != 'bwd':
        if args.verbose:
            print(' - perform {:d}-nn source against target'.format(args.neighborhood))
        x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], args.neighborhood), args.gpu, mem=args.mem * 1024 ** 3)
        x2y_mean = x2y_sim.mean(axis=1)
        print(f' - {x2y_mean=}')
        del x2y_sim

    if args.retrieval != 'fwd':
        if args.verbose:
            print(' - perform {:d}-nn target against source'.format(args.neighborhood))
        y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], args.neighborhood), args.gpu, mem=args.mem * 1024 ** 3)
        y2x_mean = y2x_sim.mean(axis=1)
        print(f' - {y2x_mean=}')
        del y2x_sim
    
    # margin function
    if args.margin == 'absolute':
        margin = lambda a, b: a
    elif args.margin == 'distance':
        margin = lambda a, b: a - b
    else:  # args.margin == 'ratio':
        margin = lambda a, b: a / b

    fout = None
    # fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    os.makedirs("/".join(args.output.split("/")[:-1]), exist_ok=True)
    filter_fn = build_filter_fn(args)

    if args.mode == 'search':
        raise ValueError
        if args.verbose:
            print(' - Searching for closest sentences in target')
            print(' - writing alignments to {:s}'.format(args.output))
        scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

        nbex = x.shape[0]
        ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
        err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
        print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))
        for i, idx in enumerate(src_ids):
            print(trg_sents[best[i]], file=fout)

    elif args.mode == 'score':
        raise ValueError
        for i, j in zip(src_inds, trg_inds):
            s = score(x[i], y[j], x2y_mean[i], y2x_mean[j], margin)
            print(s, src_sents[i], trg_sents[j], sep='\t', file=fout)

    elif args.mode == 'mine':
        if args.verbose:
            print(' - mining for parallel data')
        fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin, args.verbose)
        bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin, args.verbose)
        fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]
        
        x_shape = x.shape
        y_shape = y.shape
        del x, y
        fsrc = ftgt = None
        src_texts = tgt_texts = None
        if args.src_txt is not None and args.tgt_txt is not None:
            print(f'load source text file {args.src_txt}')
            with open(args.src_txt, 'r', encoding=args.encoding, errors='surrogateescape') as f:
                # src_texts = f.read().strip().split('\n')
                src_texts = f.read().splitlines()
            
            print(f'load target text file {args.tgt_txt}')
            with open(args.tgt_txt, 'r', encoding=args.encoding, errors='surrogateescape') as f:
                # tgt_texts = f.read().strip().split('\n')
                tgt_texts = f.read().splitlines()
            
            fsrc = open(f'{args.output}.txt.{args.src_lang}', mode='w', encoding=args.encoding, errors='surrogateescape')
            ftgt = open(f'{args.output}.txt.{args.tgt_lang}', mode='w', encoding=args.encoding, errors='surrogateescape')

        align_src_idx = []
        align_tgt_idx = []
        align_scores = []
        if args.verbose:
            print(' - writing alignments to {:s}'.format(args.output))
            if args.threshold > 0:
                print(' - with threshold of {:f}'.format(args.threshold))

        accept_count = 0
        if args.retrieval == 'fwd':
            for i, j in enumerate(fwd_best):
                # print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
                accept_count += filter_fn(fwd_scores[i].max(), src_ids[i], tgt_ids[j], align_src_idx, align_tgt_idx, align_scores, src_texts, tgt_texts, fsrc, ftgt)
        if args.retrieval == 'bwd':
            for j, i in enumerate(bwd_best):
                # print(bwd_scores[j].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
                accept_count += filter_fn(bwd_scores[j].max(), src_ids[i], tgt_ids[j], align_src_idx, align_tgt_idx, align_scores, src_texts, tgt_texts, fsrc, ftgt)
        if args.retrieval == 'intersect':
            for i, j in enumerate(fwd_best):
                if bwd_best[j] == i:
                    # print(fwd_scores[i].max(), src_sents[i], trg_sents[j], sep='\t', file=fout)
                    accept_count += filter_fn(fwd_scores[i].max(), src_ids[i], tgt_ids[j], align_src_idx, align_tgt_idx, align_scores, src_texts, tgt_texts, fsrc, ftgt)
        if args.retrieval == 'max':
            indices = np.stack((np.concatenate((np.arange(x_shape[0]), bwd_best)),
                                np.concatenate((fwd_best, np.arange(y_shape[0])))), axis=1)
            scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
            seen_src, seen_tgt = set(), set()
            for i in np.argsort(-scores):
                src_ind, tgt_ind = indices[i]
                if not src_ind in seen_src and not tgt_ind in seen_tgt:
                    seen_src.add(src_ind)
                    seen_tgt.add(tgt_ind)
                    if scores[i] > args.threshold:
                        # print(scores[i], src_sents[src_ind], trg_sents[trg_ind], sep='\t', file=fout)
                        accept_count += filter_fn(scores[i], src_ids[src_ind], tgt_ids[tgt_ind], align_src_idx, align_tgt_idx, align_scores, src_texts, tgt_texts, fsrc, ftgt)

        if fsrc is not None and ftgt is not None:
            fsrc.close()
            ftgt.close()
        save_obj = {
            'src_idx': np.array(align_src_idx),
            'tgt_idx': np.array(align_tgt_idx),
            'scores': np.array(align_scores),
            'x2y_mean': x2y_mean, 
            'y2x_mean': y2x_mean,
        }
        torch.save(save_obj, args.output)
        print(f'Accept: {accept_count} / {len(src_ids)} / {len(tgt_ids)} ({accept_count / float(len(src_ids))})')

    # fout.close()


