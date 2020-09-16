import os.path as op
import sys

sys.path.insert(0, '/private/home/changhan/projects/fairseq_s2t/fb_sweep')

import sweep
from sweep import hyperparam as hp


PT_MODEL_ROOT = '/private/home/changhan/data/models'
MBART_PATH = f'{PT_MODEL_ROOT}/mbart/MBART_finetuned_enro/model_upgraded.pt'
W2V_PATH = f'{PT_MODEL_ROOT}/wav2vec2/wav2vec_small.pt'

MANIFEST_ROOT = '/private/home/changhan/data/datasets/speech/must_c_1_0'
LANG = 'ro'
MANIFEST_ID = f'ast_{LANG}_w2v_aud'

SPM_FILE = 'sentence.bpe.model'


def get_filename_wo_ext(path):
    return op.splitext(op.basename(path))[0]


FIXED_LR_SCHEDULE = [
    hp('--lr', 1e-4),
    hp('--lr-scheduler', 'fixed'),
]

LR_SCHEDULE = [
    hp('--lr', 5e-4, save_dir_key=lambda v: f'lr_{v}'),
    hp('--lr-scheduler', 'inverse_sqrt'),
    hp('--warmup-updates', 5000),
]


def get_grid(args):
    return [
               hp(f'{MANIFEST_ROOT}/en-ro/manifests/ast_ro_w2v_aud',
                  positional_arg=True),
               hp('--bpe', 'sentencepiece'),
               hp(
                   '--sentencepiece-model',
                   f'{MANIFEST_ROOT}/en-{LANG}/manifests/{MANIFEST_ID}/{SPM_FILE}',
               ),
               hp('--train-subset', 'train'),
               hp('--valid-subset', 'dev'),

               hp('--ddp-backend', 'no_c10d'),
               hp('--num-workers', 1),

               hp('--task', 'speech_to_text'),
               hp('--criterion', 'label_smoothed_cross_entropy_with_accuracy'),
               hp('--label-smoothing', 0.1, save_dir_key=lambda v: f'ls_{v}'),
               hp('--arch', 'xm_transformer'),
               hp('--max-tokens', 160 * 10 * 16),
               hp('--max-sentences', 1),
               hp('--max-tokens-valid', 5000 * 10 * 16),

               hp('--optimizer', 'adam'),
               hp('--clip-norm', 10.0),

               hp('--log-format', 'simple'),
               hp('--log-interval', 50),
               hp('--keep-last-epochs', 10),
               hp('--fp16'),
               hp('--seed', 1),

               hp('--w2v-path', W2V_PATH,
                  save_dir_key=lambda v: f'enc_pt_{get_filename_wo_ext(v)}'),
               hp('--load-pretrained-decoder-from', MBART_PATH,
                  save_dir_key=lambda v: f'dec_pt_{get_filename_wo_ext(v)}'),

               hp('--max-update', 300000),
               hp('--update-freq', 8, save_dir_key=lambda v: f'up_freq_{v}'),

               hp('--skip-invalid-size-inputs-valid-test'),

               hp('--save-interval', 5),
           ] + LR_SCHEDULE


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
