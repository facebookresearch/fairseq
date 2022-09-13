#!/usr/bin/env python


from fb_sweep import sweep
from fb_sweep.sweep import hyperparam as hp
import os.path as op
import os

LP7  = sorted(
    [
        "ph-es",
        "ph-fr",
        "ph-it",
    ]
)
#bitext batch number is 1042201 
#sup_speech_ali batch number is 114762 
#sup_speech_s2s batch number is 371118 
#unsup_speech batch number is 3545372 

#sup:250k
#sups2s:250k
#unsup_speech: 3545k
#bitext: 1666k
dataschem={
   "R3": {     
        "mu": 1600000,
        "sup": "3",
        "sups2s": "3",
        "supent": 10,
        "unsup": "1", 
        "text": "1",
        "bitext": "1",
   },
}
ds='R3'
TRAIN_SUP='train_mustc_ph_it_align,train_ep_ph_it_align,train_mustc_ph_es_align,train_ep_ph_es_align,train_mustc_ph_fr_align,train_ep_ph_fr_align'
VALID_SUP='dev_ep_ph_it_align,dev_ep_ph_fr_align,dev_ep_ph_es_align'
SUP_DIR='/large_experiments/ust/mgaido/2022/data/joint_s2t/st/'

SUP_DIR_S2S='/large_experiments/ust/mgaido/2022/data/joint_s2t/st/'
TRAIN_SUP_S2S='train_mustc_ph_it_st,train_ep_ph_it_st,train_mustc_ph_es_st,train_ep_ph_es_st,train_mustc_ph_fr_st,train_ep_ph_fr_st,train_mustc_ph_es_asr,train_ep_ph_es_asr'
VALID_SUP_S2S='dev_ep_ph_it_st,dev_ep_ph_fr_st,dev_ep_ph_es_st'

SUP_DIR_ENTITIES='/large_experiments/ust/mgaido/2022/data/joint_s2t/st/'
TRAIN_SUP_ENTITIES='train_mustc_ph_es_st_with_entities,train_ep_ph_es_st_with_entities'
VALID_SUP_ENTITIES='dev_ep_ph_es_st_with_entities'


UNSUP_TRAIN='/private/home/yuntang/2020/T70609114_st_joint_train/wav2vec/outputs/data/train.tsv' #'/checkpoint/changhan/voxpopuli_100k/manifest_wo_asr_eval.tsv'
UNSUP_VALID='/private/home/yuntang/2020/T70609114_st_joint_train/wav2vec/outputs/data/valid_s.tsv'
bart='/checkpoint/mgaido/2022/ne/txt_pretr.xent.wd0.01.uf_16.neuu.oprj.config.lpbph-es_ph-fr_ph-it.lpph-en.lr_0.001.elr_1e-06.mu1600.0k.bs100.msp512.mtp512.mtt1024.msa0.7.lm.ttkn.mwd.mask0.3.mr0.1.ssmp0.3.sump0.7.s2s.gelu.default.nb.lpos.lne.dp0.1.ngpu8/best_bart.pt'
weight_decay = 0.01


def get_fairseq_root():
    fairseq_root=os.getenv('fairseq_root') 
    if fairseq_root is None:
        fairseq_root = '/private/home/mgaido/fairseq'
    return fairseq_root

def get_filename_wo_ext(path):
    return op.splitext(op.basename(path))[0]


LR_SCHEDULE = [
    hp('--lr', [ 0.001 ], save_dir_key=lambda v: f'lr_{v}'),
    hp('--end-learning-rate', [ 1e-6], save_dir_key=lambda v: f'elr_{v}'),
    hp('--lr-scheduler', 'polynomial_decay'),
    hp('--warmup-updates', 10000),
    hp('--total-num-update', dataschem[ds]['mu'], save_dir_key=lambda v:f'mu{v/1000}k'),   
    hp('--validate-interval-updates', 10000),
]

DATA_SETTING = [
     hp('--train-subset', 'train'),
     hp('--valid-subset', 'valid_bitext,valid_sup_speech,valid_sup_speech_s2s,valid_sup_speech_entities,valid_unsup_speech'),   #
     hp('--dataset-impl', 'mmap'),
     hp('--sup-speech-data', SUP_DIR),
     hp('--sup-speech-train-subset', TRAIN_SUP),
     hp('--sup-speech-valid-subset', VALID_SUP),
     hp('--sup-speech-s2s-data', SUP_DIR_S2S),
     hp('--sup-speech-s2s-train-subset', TRAIN_SUP_S2S),
     hp('--sup-speech-s2s-valid-subset', VALID_SUP_S2S),
     hp('--sup-speech-entities-data', SUP_DIR_ENTITIES),
     hp('--sup-speech-entities-train-subset', TRAIN_SUP_ENTITIES),
     hp('--sup-speech-entities-valid-subset', VALID_SUP_ENTITIES),
     hp('--unsup-speech-train-data', UNSUP_TRAIN),
     hp('--unsup-speech-valid-data', UNSUP_VALID),
]
DATA_HYP = [
     hp('--batch-size', 100, save_dir_key=lambda v:f'bs{v}'),
     hp('--batch-size-valid', 100),
     hp('--max-source-positions', 512, save_dir_key=lambda v:f'msp{v}'),
     hp('--max-target-positions', 512, save_dir_key=lambda v:f'mtp{v}'),

     hp('--max-text-tokens', 1024, save_dir_key=lambda v:f'mtt{v}'),
     hp('--max-speech-positions', 600000, ),
     hp('--max-sample-size', 600000, ),
     hp('--min-sample-size', 32000, ),
     hp('--max-speech-tokens', 600000, ),
     hp('--skip-invalid-size-inputs-valid-test'),
     hp('--multilang-sampling-alpha', [0.7], save_dir_key=lambda v:f"msa{v}"),

     #ml5
     hp('--unsupervised-speech-sample-ratio', dataschem[ds]['unsup'] , save_dir_key=lambda v:f'{ds}'),
     hp('--supervised-speech-sample-ratio', dataschem[ds]['sup'], ),
     hp('--supervised-speech-s2s-sample-ratio', dataschem[ds]['sups2s'],),
     hp('--supervised-speech-entities-sample-ratio', dataschem[ds]['supent'],),
     ###ml4
     hp('--text-sample-ratio', dataschem[ds]['text'] ),
     hp('--bitext-sample-ratio', dataschem[ds]['bitext'] ),
     hp("--add-tgt-lang-token", save_dir_key=lambda x: "ttkn"), 
     hp("--use-mask-whole-words", save_dir_key=lambda x: "mwd"),
]
MASKING = [
    hp('--mask', 0.3, save_dir_key=lambda v:f'mask{v}'),
    hp('--mask-random', 0.1, save_dir_key=lambda v:f'mr{v}'),
    hp('--mask-length', 'span-poisson'),
    hp('--speech-sup-mask-prob', [0.3], save_dir_key=lambda v:f'ssmp{v}'),
    hp('--speech-unsup-mask-prob', [0.7], save_dir_key=lambda v:f'sump{v}'),
]
MODEL = [
    hp('--arch', 'speech_text_pretrain_bart_base_stack'),
    hp('--no-scale-feature' ),
    hp('--stacked-encoder', ['s2s'], save_dir_key=lambda v: v),
    hp('--activation-fn', 'gelu', save_dir_key=lambda v: v),
    hp('--speech-extractor-mode', 'default', save_dir_key=lambda v:v),       
    hp('--encoder-normalize-before', save_dir_key=lambda v: 'nb'),
    hp('--decoder-normalize-before', ),
    # mbart setting 
    hp('--encoder-learned-pos', save_dir_key=lambda v: 'lpos'),
    hp('--decoder-learned-pos'),
    hp('--layernorm-embedding', save_dir_key=lambda v: 'lne'),
    hp('--share-decoder-input-output-embed', save_dir_key=lambda v: 'shemb'),

    # wav2vec
    hp('--dropout', [0.1], save_dir_key=lambda v:f'dp{v}'),
    hp('--load-pretrained-mbart-encoder-from', bart, save_dir_key=lambda v: "bart"),
    hp('--load-pretrained-mbart-decoder-from', bart, ),

    # entity retrieval
    hp('--er-modality-embedding'),
    hp('--er-window-attention-mask'),
]
def get_grid(args):

    args.data =  "/large_experiments/ust/mgaido/2022/data/joint_s2t/txt/"
    return [
               hp('--user-dir', f'{get_fairseq_root()}/examples/speech_text_joint_to_text'),
               hp('--task', 'speech_text_joint_denoising'),
               hp('--criterion', 'speech_text_pretrain_cross_entropy', save_dir_key=lambda val: 'xent'),
               #hp('--label-smoothing', 0.1, save_dir_key=lambda val: f'ls{val}'),
               hp("--optimizer", "adam" ),
               #hp("--grad-proj-deduction-ratio", [0.0], save_dir_key=lambda val: f"gpr{val}"),
               hp("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
               hp('--update-freq', 8, save_dir_key=lambda v: f'uf_{v}'),

               hp('--no-emb-update-unsup', save_dir_key=lambda v:"neuu"),
               hp('--use-decoder-output-proj', save_dir_key=lambda v:"oprj"),

               hp('--config-yaml', 'config_ph.yaml', save_dir_key=lambda v: f'{get_filename_wo_ext(v)}'),
               hp('--config-s2s-yaml', 'config_st.yaml'), #from different directories
               hp('--config-entities-yaml', 'config_st.yaml'), #from different directories
               hp('--ddp-backend', 'no_c10d'),
               hp('--lang-pairs-bitext', ",".join(LP7), save_dir_key=lambda v: 'lp' + v),
               hp('--lang-pairs', "ph-en", save_dir_key=lambda v: 'lp' + v),
               hp('--num-workers', 2),
               hp('--log-interval', 500),
               hp('--save-interval-updates', 10000),
               hp('--keep-interval-updates', 1),
               hp('--report-accuracy'),
               hp('--max-sentences', 20),

           ] + LR_SCHEDULE + DATA_SETTING + DATA_HYP + MASKING + MODEL


def postprocess_hyperparams(args, config):
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)

'''
py ../examples/speech_translation/sweep_xmtl.py \
    --checkpoints-dir /checkpoint/changhan/projects/fairseq_s2t_xm \
    -p xm -t -1 -g 8 -n 1 --mem 500G \
    --script /private/home/changhan/projects/fairseq_s2t/train.py \
    --python /private/home/changhan/py_fairseq_s2t/bin/python \
    --backend slurm --partition learnfair --constraint volta32gb \
    --comment xmtl
'''
