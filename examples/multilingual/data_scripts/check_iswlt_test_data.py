# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os, sys
import subprocess
import re
from subprocess import check_call, check_output

WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)


BLEU_REGEX = re.compile("^BLEU\\S* = (\\S+) ")
def run_eval_bleu(cmd):
    output = check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
    print(output)
    bleu = -1.0
    for line in output.strip().split('\n'):
        m = BLEU_REGEX.search(line)
        if m is not None:
            bleu = m.groups()[0]
            bleu = float(bleu)
            break
    return bleu

def check_data_test_bleu(raw_folder, data_lang_pairs):
    not_matchings = []
    for sacrebleu_set, src_tgts in data_lang_pairs:
        for src_tgt in src_tgts:
            print(f'checking test bleus for: {src_tgt} at {sacrebleu_set}')
            src, tgt = src_tgt.split('-')
            ssrc, stgt = src[:2], tgt[:2]
            if os.path.exists(f'{raw_folder}/test.{tgt}-{src}.{src}'):
                # reversed direction may have different test set
                test_src = f'{raw_folder}/test.{tgt}-{src}.{src}'
            else:
                test_src = f'{raw_folder}/test.{src}-{tgt}.{src}'
            cmd1 = f'cat {test_src} | sacrebleu -t "{sacrebleu_set}" -l {stgt}-{ssrc}; [ $? -eq 0 ] || echo ""'
            test_tgt = f'{raw_folder}/test.{src}-{tgt}.{tgt}'       
            cmd2 = f'cat {test_tgt} | sacrebleu -t "{sacrebleu_set}" -l {ssrc}-{stgt}; [ $? -eq 0 ] || echo ""'
            bleu1 = run_eval_bleu(cmd1) 
            if bleu1 != 100.0:
                not_matchings.append(f'{sacrebleu_set}:{src_tgt} source side not matching: {test_src}')
            bleu2 = run_eval_bleu(cmd2) 
            if bleu2 != 100.0:
                not_matchings.append(f'{sacrebleu_set}:{src_tgt} target side not matching: {test_tgt}')
    return not_matchings       

if __name__ == "__main__":
    to_data_path = f'{WORKDIR_ROOT}/iwsltv2'
    not_matching = check_data_test_bleu(
        f'{to_data_path}/raw', 
        [
            ('iwslt17', ['en_XX-ar_AR', 'en_XX-ko_KR', 'ar_AR-en_XX', 'ko_KR-en_XX']),
            ('iwslt17', ['en_XX-it_IT', 'en_XX-nl_XX', 'it_IT-en_XX', 'nl_XX-en_XX']),
            ('iwslt17/tst2015', ['en_XX-vi_VN', "vi_VN-en_XX"]),        
        ]
        )    
    if len(not_matching) > 0:
        print('the following datasets do not have matching test datasets:\n\t', '\n\t'.join(not_matching))

