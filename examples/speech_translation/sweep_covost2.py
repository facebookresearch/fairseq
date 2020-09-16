import os
import sys

sys.path.insert(0, '/private/home/changhan/projects/fairseq_s2t/fb_sweep')

import sweep
from sweep import hyperparam as hp

HIRES_XX_EN_LANGS = ['fr', 'de', 'es', 'ca', 'it', 'ru', 'zh-CN', 'pt', 'fa']
LORES_XX_EN_LANGS = ['et', 'mn', 'nl', 'tr', 'ar', 'sv-SE', 'lv', 'sl', 'ta',
                     'ja', 'id', 'cy']
XX_EN_LANGS = HIRES_XX_EN_LANGS + LORES_XX_EN_LANGS
EN_XX_LANGS = ['de', 'tr', 'fa', 'sv-SE', 'mn', 'zh-CN', 'cy', 'ca', 'sl', 'et',
               'id', 'ar', 'ta', 'lv', 'ja']
DATA_ROOT = '/private/home/changhan/data/datasets/speech/common_voice_20191210'

TASK = os.environ.get('TASK', 'en_asr')
MODEL = os.environ.get('MODEL', 't')

TASK_TO_SRC = {
    'en_asr': 'en', 'all_ast_en': 'all_en', 'all_ast_all': 'all_all'
}
for lang in XX_EN_LANGS:
    TASK_TO_SRC[f'{lang}_ast_en'] = lang
for lang in EN_XX_LANGS:
    TASK_TO_SRC[f'en_ast_{lang}'] = 'en'
SRC = TASK_TO_SRC.get(TASK)

TASK_TO_TGT = {
    'en_asr': 'asr_char', 'all_ast_en': 'ast_en_char', 'all_ast_all': 'ast_char'
}
for lang in XX_EN_LANGS:
    TASK_TO_TGT[f'{lang}_ast_en'] = 'ast_en_char'
for lang in EN_XX_LANGS:
    TASK_TO_TGT[f'en_ast_{lang}'] = f'ast_{lang}_char'
TGT = TASK_TO_TGT.get(TASK)

TRANSFORMER_S = [
    hp('--arch', 's2t_transformer_s', save_dir_key=lambda v: f'arch_{v}'),
    hp('--dropout', 0.1, save_dir_key=lambda v: f'dp_{v}'),
    hp('--optimizer', 'adam'),
    hp('--adam-betas', '(0.9, 0.98)'),
    hp('--lr', [1e-3], save_dir_key=lambda v: f'lr_{v}'),
    hp('--lr-scheduler', 'inverse_sqrt'),
    hp('--warmup-updates', 10000, save_dir_key=lambda v: f'wu_{v}')
]

TRANSFORMER_M = [
    hp('--arch', 's2t_transformer_m', save_dir_key=lambda v: f'arch_{v}'),
    hp('--dropout', 0.15, save_dir_key=lambda v: f'dp_{v}'),
    hp('--optimizer', 'adam'),
    hp('--adam-betas', '(0.9, 0.98)'),
    hp('--lr', [1e-3], save_dir_key=lambda v: f'lr_{v}'),
    hp('--lr-scheduler', 'inverse_sqrt'),
    hp('--warmup-updates', 10000, save_dir_key=lambda v: f'wu_{v}')
]

BERARD = [
    hp('--arch', 'berard_512_5_3', save_dir_key=lambda v: f'arch_{v}'),
    hp('--optimizer', 'adam'),
    hp('--lr-scheduler', 'fixed'),
    hp('--lr', 1e-3, save_dir_key=lambda v: f'lr_{v}'),
]

EN_ASR = [
    hp('--train-subset', 'train_covost'),
    hp('--valid-subset', 'dev_cv'),
    hp('--max-tokens', 50000),
    hp('--validate-interval', 5),
    hp('--max-update', 60000),
]
CHECKPOINT_ROOT = '/checkpoint/changhan/projects/fairseq_s2t'
T_S_EN_ASR_CHKPT = f'{CHECKPOINT_ROOT}/covost2.task_en_asr_char.' \
                   'ls_0.1.arch_s2t_transformer_s.dp_0.1.lr_0.001.wu_10000.ngpu8/' \
                   'avg_last_10_checkpoint.pt'
T_M_EN_ASR_CHKPT = f'{CHECKPOINT_ROOT}/covost2.task_en_asr_char.' \
                   f'ls_0.1.arch_s2t_transformer_m.dp_0.15.lr_0.001.wu_10000.ngpu8/' \
                   'avg_last_10_checkpoint.pt'

EN_ASR_CHKPT = {'t_s': T_S_EN_ASR_CHKPT, 't_m': T_M_EN_ASR_CHKPT}.get(MODEL)

ST = [
    hp('--train-subset', 'train_covost'),
    hp('--valid-subset', 'dev_cv'),
    hp('--validate-interval', 5),
    hp('--load-pretrained-encoder-from', EN_ASR_CHKPT,
       save_dir_key=lambda v: 'ptenc'),
]
HIRES_ST = ST + [
    hp('--max-tokens', 40000),
    hp('--max-update', 60000),
]
LORES_ST = ST + [
    hp('--max-tokens', 5000),
    hp('--max-update', 60000),
]
# ast all-en
ALL_AST_EN = [
    hp('--train-subset', ','.join(f'train_{x}' for x in XX_EN_LANGS)),
    hp('--valid-subset', ','.join(f'dev_{x}' for x in XX_EN_LANGS)),
    hp('--config-yaml', 'config_sa0.5.yaml',
               save_dir_key=lambda v: f'cfg_{v}'),
    hp('--max-tokens', 35000),
    hp('--validate-interval', 5),
    hp('--load-pretrained-encoder-from', EN_ASR_CHKPT,
       save_dir_key=lambda v: 'ptenc'),
    hp('--max-update', 150000),
]

# ast all-all
ALL_AST_ALL = [
    hp(
        '--train-subset',
        ','.join([f'train_{x}_en' for x in XX_EN_LANGS] +
                 [f'train_en_{x}' for x in EN_XX_LANGS])
    ),
    hp(
        '--valid-subset',
        ','.join([f'dev_{x}_en' for x in XX_EN_LANGS] +
                 [f'dev_en_{x}' for x in EN_XX_LANGS])
    ),
    hp('--config-yaml', 'config_sa0.5.yaml', save_dir_key=lambda v: f'cfg_{v}'),
    hp('--max-tokens', 40000),  # 2 nodes
    hp('--validate-interval', 5),
    hp('--load-pretrained-encoder-from', EN_ASR_CHKPT,
       save_dir_key=lambda v: 'ptenc'),
    hp('--max-update', 200000),
]

TASK_TO_TASK_SPECIFIC = {
    'en_asr': EN_ASR, 'all_ast_en': ALL_AST_EN, 'all_ast_all': ALL_AST_ALL
}
for lang in HIRES_XX_EN_LANGS:
    TASK_TO_TASK_SPECIFIC[f'{lang}_ast_en'] = HIRES_ST
for lang in LORES_XX_EN_LANGS:
    TASK_TO_TASK_SPECIFIC[f'{lang}_ast_en'] = LORES_ST
TASK_SPECIFIC = TASK_TO_TASK_SPECIFIC[TASK]
MODEL_SPECIFIC = {
    't_s': TRANSFORMER_S, 't_m': TRANSFORMER_M, 'b': BERARD
}[MODEL]


def get_grid(args):
    return [
               hp(f'{DATA_ROOT}/{SRC}/manifests/{TGT}', positional_arg=True),

               hp('--bpe', 'sentencepiece'),
               hp('--sentencepiece-model',
                  f'{DATA_ROOT}/{SRC}/manifests/{TGT}/spm_char.model',
                  save_dir_key=lambda v: f'task_{TASK}_char'),

               hp('--ddp-backend', 'no_c10d'),
               hp('--num-workers', 10),

               hp('--task', 'speech_to_text'),
               hp('--criterion', 'label_smoothed_cross_entropy_with_accuracy'),
               hp('--label-smoothing', 0.1, save_dir_key=lambda v: f'ls_{v}'),
               hp('--max-tokens-valid', 50000),
               hp('--clip-norm', 10.0),

               hp('--log-format', 'simple'),
               hp('--log-interval', 50),
               hp('--keep-last-epochs', 10),
               hp('--fp16'),
               hp('--seed', 1),

               hp('--input-feat-per-channel', 80)
] + MODEL_SPECIFIC + TASK_SPECIFIC


def postprocess_hyperparams(args, config):
    if config['--max-tokens'].current_value == 5000 \
            and config['--arch'].current_value == 's2t_transformer_s':
        config['--lr'].current_value = 2e-4


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)

'''
for LANG in et mn nl tr ar sv-SE lv sl ta ja id cy; do
for LANG in fr de es ca it ru zh-CN pt fa; do
TASK=${LANG}_ast_en MODEL=t_s py ../examples/speech_translation/sweep_covost2.py \
    --comment eacl --checkpoints-dir /checkpoint/changhan/projects/fairseq_s2t \
    -p covost2 -t -1 -g 8 -n 1 --mem 500G \
    --script /private/home/changhan/projects/fairseq_s2t/train.py \
    --python /private/home/changhan/py_fairseq_s2t/bin/python \
    --backend slurm --partition priority --constraint volta32gb
done
'''
