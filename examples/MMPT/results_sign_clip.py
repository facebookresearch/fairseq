from pathlib import Path
import subprocess
import argparse
import yaml
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overwrite', required=False, action='store_true', help='whether to overwrite existing testing logs')
args = parser.parse_args()

tasks_path = 'projects/retri/signclip_v1/'
tasks = [
    'baseline',
    'baseline_aug',
    'baseline_unique',
    'baseline_asl',
    'baseline_asl_finetune',
    'baseline_asl_plus',
    'baseline_asl_plus_fs',
    # 'baseline_sp_b768_zs'
    'baseline_sp_b768_finetune_asl',
    'baseline_sp_b768_pre_finetune_asl',
]
notes = [
    'baseline (asl_signs data)',
    'baseline with 2d augmentation',
    'baseline with unique sampler',
    'baseline trained with 3 ASL ISLR datasets',
    'baseline trained with 3 ASL ISLR datasets then finefuned',
    'baseline trained with 3 ASL ISLR datasets (including validation data)',
    'baseline trained with 3 ASL ISLR datasets + ChicagoFS (including validation data)',
    # 'baseline trained with Spreadthesign then zero-shot',
    'baseline trained with Spreadthesign then finefuned',
    'baseline trained with Spreadthesign (sign-vq preprocessing) then finefuned',
]

results = {}
for task in tasks:
    task_path = tasks_path + task + '.yaml'
    config = yaml.safe_load(Path(task_path).read_text())
    log_path = f"{config['eval']['save_path']}/test.log"
    command = f'python locallaunch.py {task_path} --jobtype local_predict'

    print(command)

    # if args.overwrite or (not Path(log_path).is_file()):
    #     Path(log_path).parent.mkdir(exist_ok=True, parents=True)
    #     with open(log_path, 'w') as f:
    #         subprocess.run(command, shell=True, stdout=f)

    result = {}
    with open(log_path, 'r') as f:
        prev_line = ''
        for line in f.readlines():
            line = line.strip()
            if prev_line.startswith('text to video'):
                prefix = 'T2V'
            elif prev_line.startswith('video to text'):
                prefix = 'V2T'
            else:
                prev_line = line
                continue

            metrics = line.split(' - ')
            for metric in metrics:
                name, value = metric.split(': ')
                result[f'{prefix}_{name}'] = value

            prev_line = line

    results[task] = result

df = pd.DataFrame.from_dict(results, orient='index')
df = df[['T2V_P@1', 'T2V_P@5', 'T2V_P@10', 'T2V_Median R', 'V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_Median R']]
df['notes'] = notes

print(df)

df.to_csv('results_sign_clip.csv')
    