"""
Usage:
    This scripts it to evaluate the classification accuracy/error rate from the embedding extracted
    by gen_audio_embedding.py 
    Example (LID classification)

    PYTHONPATH='.' python examples/wav2vec/eval_speaker_clf_task.py \
            --data /fsx/androstj/exps/lid_voxlingua/infer/atj_xlsr2_100pct_300M_mean_fast_upd_100k_new.npz \
            --task cls --merge mean_logit
"""
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import ipdb
import logging
import argparse
from scipy.special import softmax

log=logging.getLogger(__name__)
log.setLevel(logging.INFO)

def calculate_eer(y_label, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y_label, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    optimal_threshold = interp1d(fpr, thresholds)(eer)
    return eer, optimal_threshold

def calculate_minDCF(y_label, y_score, p_target=0.01, c_miss=1, c_fa=1):
    # https://github.com/kaldi-asr/kaldi/blob/master/egs/sre08/v1/sid/compute_min_dcf.py
    from sklearn.metrics import det_curve
    fpr, fnr, thresholds = det_curve(y_label, y_score, pos_label=1)
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fpr)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='npz contains name & latent file')
    parser.add_argument('--task', choices=['cls', 'veri', 'cls_voxlingua'])
    parser.add_argument('--merge', choices=['mean_logit', 'first_logit', 'mean_latent_sim', 'first_latent_sim', 'mean_logit_sim', 'first_logit_sim'])
    parser.add_argument('--veri-pair', help='verification file contains 1/0 utt_x utt_y')
    parser.add_argument('--scaler', type=str, choices=['mean_var'])
    parser.add_argument('--compress-method', choices=['pca'])
    parser.add_argument('--compress-dim', type=int)
    args = parser.parse_args()

    if args.task in ['cls', 'cls_voxlingua']:
        print('| run classification evaluation')
        data = np.load(args.data)
        data_logit = data['logit']
        data_target = data['target']
        data_src_len = data['src_len']
        assert data_logit.shape[0] ==  data_target.shape[0]
        B = data_logit.shape[0]
        correct = 0
        total = 0
        data_prob = softmax(data_logit, axis=2)
        correct_vs_len = np.empty((B, 2))
        for ii in range(B):
            _target = data_target[ii]
            if args.merge == 'mean_logit':
                _prob = np.mean(data_prob[ii], axis=0)
                top_1 = np.argmax(_prob)
            elif args.merge == 'first_logit':
                _prob = data_prob[ii][0]
                top_1 = np.argmax(_prob)
            else :
                raise ValueError()
            is_top_1 = (1 if top_1 == _target else 0)
            correct += is_top_1
            total += 1
            _src_len = data_src_len[ii] / 16000
            correct_vs_len[ii] = [is_top_1, _src_len]

        acc = correct / total * 100
        t_5 = correct_vs_len[:, 1] <= 5
        t_20 = correct_vs_len[:, 1] > 5
        c_5 = correct_vs_len[t_5, 0].sum()
        c_20 = correct_vs_len[t_20, 0].sum()
        t_5 = t_5.sum()
        t_20 = t_20.sum()
        acc_5 = c_5 / t_5 * 100
        acc_20 = c_20 / t_20 * 100
        print(f'| acc = {acc:.2f}% -- err = {100-acc:.2f}% -- {correct=} {total=}')
        print(f'| acc 0to5 = {acc_5:.2f}% -- err = {100-acc_5:.2f}% -- {c_5=} {t_5=}')
        print(f'| acc 5to20 = {acc_20:.2f}% -- err = {100-acc_20:.2f}% -- {c_20=} {t_20=}')

        

    if args.task == 'veri':
        print('| run verification evaluation')
        veri_pairs = []
        with open(args.veri_pair) as ff:
            for fi in ff:
                a,b,c = fi.split()
                a = int(a)
                veri_pairs.append([a,b,c])
        
        data = np.load(args.data)
        if 'logit' in args.merge:
            data_latent = data['logit']
        elif 'latent' in args.merge:
            data_latent = data['latent']
        else :
            raise ValueError()

        data_name  = data['name']
        assert len(data_name) == len(data_latent)
        map_name_latent = {}

        from sklearn.pipeline import make_pipeline
        pipe = []
        if args.scaler == 'mean_var':
            print(f'| apply StandardScaler')
            pipe.append(StandardScaler())

        if args.compress_method == 'pca':
            n_comp = args.compress_dim
            print(f'| apply PCA with {n_comp=}')
            from sklearn.decomposition import PCA
            pipe.append(PCA(n_components=n_comp))
        if len(pipe) > 0 :
            pipe = make_pipeline(*pipe)
            data_latent_2d = data_latent.reshape(-1, data_latent.shape[-1])
            pipe.fit(data_latent_2d)
            data_latent_2d = pipe.transform(data_latent_2d)
            data_latent = data_latent_2d.reshape(data_latent.shape[0], data_latent.shape[1], -1)

        for ii in range(len(data_name)):
            map_name_latent[data_name[ii]] = data_latent[ii]
        labels = []
        scores = []
        for lbl, pair_a, pair_b in tqdm(veri_pairs):
            labels.append(lbl)
            pair_a = map_name_latent[pair_a]
            pair_b = map_name_latent[pair_b]
            assert pair_a.ndim == pair_b.ndim == 2
            score = cosine_similarity(pair_a, pair_b)
            if args.merge.startswith('mean'):
                score = np.mean(score)
            elif args.merge.startswith('first'):
                score = score[0, 0]
            else :
                raise ValueError()
            scores.append(score)
        labels = np.array(labels)
        scores = np.array(scores)
        eer, eer_threshold = calculate_eer(labels, scores)
        minDCF, minDCF_threshold = calculate_minDCF(labels, scores)
        print('='*40)
        print(f'| EER = {eer*100:.2f}%\tthreshold = {eer_threshold:.2f}')
        print(f'| minDCF = {minDCF:.2f}\tthreshold = {minDCF_threshold:.2f}')


