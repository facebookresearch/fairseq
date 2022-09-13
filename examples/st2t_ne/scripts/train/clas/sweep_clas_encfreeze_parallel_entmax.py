#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam
import re, os
import os.path as op

BASEDIR=f'/checkpoint/mgaido/2022/ne/'
MDLS={
    "asrst_base": {'path': "/checkpoint/mgaido/2022/ne/asrstmulti_fn.dualinputs2twavtransformer_base_stack.adam.lr0.0003.uf4.cn10.0.wu20000.gal0.8.car0.02.egm2.0.ls0.1.mst8000.0k.msp660.0k.mtt8.192k.03_f1_ep30.mcl64.mcp0.5.mcl10.mcp0.65.ssr0.15.mtr0.0.random.ticr0.5.config_st.lne.ngpu8/avg10.pt", 'act':'gelu', 'sem': 'default', 'msp': 512, 'mtp': 512, 'lne': True, 'lpos': True },
}
mdl="asrst_base"

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
        hyperparam("--task", "speech_text_context"),
        hyperparam('--arch', ['clas_dualinputs2twavtransformer_base_stack'], save_dir_key=lambda val: val), #need small learning rate
        hyperparam('--user-dir', f'{get_fairseq_root()}/examples/speech_text_joint_to_text'),

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

        hyperparam('--criterion', ['label_smoothed_cross_entropy', ]),
        hyperparam('--enc-grad-mult', [ 2.0, ], save_dir_key=lambda val: f'egm{val}'),

        hyperparam('--label-smoothing', 0.1, save_dir_key=lambda v: f'ls{v}'),
        hyperparam('--max-tokens',  800000, save_dir_key=lambda val: f'mst{val/100}k'),
        hyperparam('--max-source-positions', 660000,save_dir_key=lambda val: f'msp{val/1000}k'),
        hyperparam('--max-tokens-text',  8192, save_dir_key=lambda val: f'mtt{val/1000}k'),
        hyperparam('--max-positions-text', 512, ),
        hyperparam('--max-sentences', 100 ),
        #hyperparam('--stacked-encoder', ['s2s'], save_dir_key=lambda v: v), 

        hyperparam('--load-init-encoder', MDLS[mdl]['path'], save_dir_key=lambda val: f"{mdl}"),
        hyperparam('--load-init-decoder', MDLS[mdl]['path'], ),

        # freezing
        hyperparam('--freeze-encoder', save_dir_key=lambda val: "frzenc"),
        #hyperparam('--freeze-pretrained-decoder', save_dir_key=lambda val: "frzdec"),
        # context
        hyperparam('--context-encoder-layers', 3, save_dir_key=lambda val: f"ctxlayers_{val}"),
        hyperparam('--context-attention-type', 'parallel', save_dir_key=lambda val: f"ctxtype_{val}"),
        #hyperparam('--add-context-gating', save_dir_key=lambda val: f"ctxgate"),
        hyperparam('--context-activation-fn', 'entmax', save_dir_key=lambda val: f"ctxattnfn_{val}"),
        hyperparam('--context-max-source-positions', 300),
        
        #masking
        hyperparam('--speech-mask-channel-length', 64, save_dir_key=lambda val: f'mcl{val}'),
        hyperparam('--speech-mask-channel-prob', 0.5, save_dir_key=lambda val: f'mcp{val}'),
        hyperparam('--speech-mask-length', 10, save_dir_key=lambda val: f'mcl{val}'),
        hyperparam('--speech-mask-prob', 0.65, save_dir_key=lambda val: f'mcp{val}'),

        hyperparam('--max-tokens-valid', 800000),
        hyperparam('--ddp-backend', 'no_c10d'),
        hyperparam('--log-interval', 100),
        hyperparam('--train-subset', 'train_ep_ph_es_asr_with_enentities,train_mustc_ph_es_asr_with_enentities,train_ep_ph_es_st_with_esentities,train_mustc_ph_es_st_with_esentities,train_ep_ph_fr_st_with_frentities,train_mustc_ph_fr_st_with_frentities,train_ep_ph_it_st_with_itentities,train_mustc_ph_it_st_with_itentities'),
        hyperparam('--valid-subset', 'dev_ep_ph_es_asr_with_enentities,dev_ep_ph_es_st_with_esentities,dev_ep_ph_fr_st_with_frentities,dev_ep_ph_it_st_with_itentities'),
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
