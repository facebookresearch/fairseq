#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam
import re, os
import os.path as op

BASEDIR=f'/checkpoint/mgaido/2022/ne/'
MDLS={
    "03_f1_ep30": {'path': "/checkpoint/mgaido/2022/ne/full_pretr.xent.wd0.01.uf_8.neuu.oprj.config_ph.lpph-es_ph-fr_ph-it.lpph-en.lr_0.001.elr_1e-06.mu1600.0k.bs100.msp512.mtp512.mtt1024.msa0.7.R3.ttkn.mwd.mask0.3.mr0.1.ssmp0.3.sump0.7.s2s.gelu.default.nb.lpos.lne.shemb.dp0.1.bart.ngpu8/checkpoint_last.pt", 'act':'gelu', 'sem': 'default', 'msp': 512, 'mtp': 512, 'lne': True, 'lpos': True },
}
mdl="03_f1_ep30"

def replace_comma(s):
    return re.sub(",","_", s)

def get_fairseq_root():
    fairseq_root=os.getenv('fairseq_root') 
    if fairseq_root is None:
        fairseq_root = '/private/home/mgaido/fairseq'
    return fairseq_root

def get_filename_wo_ext(path):
    return op.splitext(op.basename(path))[0]


def get_opts():
    OPTS=[]

    if MDLS[mdl].get('lne', False):
        OPTS=OPTS + [hyperparam('--layernorm-embedding', save_dir_key=lambda v: 'lne') ]

    if MDLS[mdl].get('lpos', False):
        OPTS=OPTS + [hyperparam('--encoder-learned-pos'), hyperparam('--decoder-learned-pos')]

    OPTS=OPTS + [hyperparam('--activation-fn', MDLS[mdl]['act'])]
    OPTS=OPTS + [hyperparam('--speech-extractor-mode', MDLS[mdl]['sem'])]
    OPTS=OPTS + [hyperparam('--max-positions-text', MDLS[mdl]['msp'])]
    OPTS=OPTS + [hyperparam('--max-target-positions', MDLS[mdl]['mtp'])]
    OPTS=OPTS + [hyperparam('--encoder-normalize-before'),hyperparam('--decoder-normalize-before')]
    return OPTS


def get_grid(args):
    """
    Replicates the `16-bit+cumul+2x lr` results from Table 1 of
    "Scaling Neural Machine Translation" (https://arxiv.org/abs/1806.00187)
    """
    args.data = "/large_experiments/ust/mgaido/2022/data/joint_s2t/st"
    ##### extra text
    return [
        hyperparam('--dataset-impl', 'mmap'),
        hyperparam('--num-workers', 4),
        hyperparam("--task", "speech_text_joint_to_text"),
        hyperparam('--arch', ['dualinputs2twavtransformer_base_stack'], save_dir_key=lambda val: val), #need small learning rate
        hyperparam('--user-dir', f'{get_fairseq_root()}/examples/speech_text_joint_to_text'),
        # optimizer


        # optimizer
        hyperparam('--max-epoch', 25),
        hyperparam('--update-mix-data', ),
        ####adam
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--lr-scheduler', ['inverse_sqrt'], ),
        hyperparam('--lr', [ 0.0003 ], save_dir_key=lambda val: f'lr{val}'),     #learn rate to s2t
        hyperparam('--update-freq', [ 4 ], save_dir_key=lambda val: f'uf{val}'),    #one machine 8 GPUs , 4 update with 10k tokens or 8 updates with 5k tokens
        hyperparam('--clip-norm', 10.0, save_dir_key=lambda val: f'cn{val}' ),
        hyperparam('--warmup-updates', [ 20000 ], save_dir_key=lambda val: f'wu{val}'),

        hyperparam('--criterion', ['guided_label_smoothed_cross_entropy_with_accuracy', ]),
        hyperparam('--guide-alpha', [ 0.8 ], save_dir_key=lambda val: f'gal{val}'),
        hyperparam('--attentive-cost-regularization', [  0.02   ], save_dir_key=lambda val: f'car{val}'),
        hyperparam('--enc-grad-mult', [ 2.0, ], save_dir_key=lambda val: f'egm{val}'),

        hyperparam('--label-smoothing', 0.1, save_dir_key=lambda v: f'ls{v}'),
        hyperparam('--max-tokens',  800000, save_dir_key=lambda val: f'mst{val/100}k'),
        hyperparam('--max-source-positions', 660000,save_dir_key=lambda val: f'msp{val/1000}k'),
        hyperparam('--max-tokens-text',  8192, save_dir_key=lambda val: f'mtt{val/1000}k'),
        hyperparam('--max-positions-text', 512, ),
        hyperparam('--max-sentences', 100 ),
        #hyperparam('--stacked-encoder', ['s2s'], save_dir_key=lambda v: v), 

        hyperparam('--load-pretrained-speech-text-encoder', MDLS[mdl]['path'], save_dir_key=lambda val: f"{mdl}"),
        hyperparam('--load-pretrained-speech-text-decoder', MDLS[mdl]['path'], ),

        #masking
        hyperparam('--speech-mask-channel-length', 64, save_dir_key=lambda val: f'mcl{val}'),
        hyperparam('--speech-mask-channel-prob', 0.5, save_dir_key=lambda val: f'mcp{val}'),
        hyperparam('--speech-mask-length', 10, save_dir_key=lambda val: f'mcl{val}'),
        hyperparam('--speech-mask-prob', 0.65, save_dir_key=lambda val: f'mcp{val}'),

        hyperparam('--text-sample-ratio', [ 0.15 ], save_dir_key=lambda val: f'ssr{val}'),
        hyperparam('--mask-text-ratio', [  0.0 ], save_dir_key=lambda val: f'mtr{val}'),
        hyperparam('--mask-text-type', ['random' ], save_dir_key=lambda val: f'{val}'),
        hyperparam('--parallel-text-data', "/large_experiments/ust/mgaido/2022/data/joint_s2t/txt/"),
        hyperparam('--text-input-cost-ratio', [  0.5  ], save_dir_key=lambda val: f'ticr{val}'),
        hyperparam('--langpairs', 'ph-es,ph-fr,ph-it'),

        hyperparam('--max-tokens-valid', 800000),
        hyperparam('--ddp-backend', 'no_c10d'),
        hyperparam('--log-interval', 100),
        hyperparam('--train-subset', 'train_mustc_ph_it_st,train_ep_ph_it_st,train_mustc_ph_es_st,train_ep_ph_es_st,train_mustc_ph_fr_st,train_ep_ph_fr_st'),
        hyperparam('--valid-subset', 'dev_ep_ph_it_st,dev_ep_ph_fr_st,dev_ep_ph_es_st'),
        hyperparam('--config-yaml', 'config_st.yaml', save_dir_key=lambda val: f'{get_filename_wo_ext(val)}'),
        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--noise-token', "▁NOISE"),

        hyperparam('--keep-last-epochs', 30),
    ]  + get_opts()



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
