from pathlib import Path
import subprocess
import argparse
import yaml
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overwrite', required=False, action='store_true', help='whether to overwrite existing testing logs')
args = parser.parse_args()

tasks_path = 'projects/retri/'
tasks = [
    'signclip_v1_1/baseline',
    'signclip_v1_1/baseline_layer',
    'signclip_v1_1/baseline_proj',
    'signclip_v1_1/baseline_pre',
    'signclip_v1_1/baseline_anonym',
    'signclip_v1_1/baseline_handedness',
]
notes = [
]

results = {}
for task in tasks:
    task_path = tasks_path + task + '.yaml'

    if not os.path.exists(task_path):
        results[task] = {}
        continue

    config = yaml.safe_load(Path(task_path).read_text())
    log_path = f"{config['eval']['save_path']}/test.log"

    if not os.path.exists(log_path):
        results[task] = {}
        continue

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

    log_path_zs = f"{config['eval']['save_path']}/test.zs.log"

    if os.path.exists(log_path_zs):
        datasets = ['AS', 'AC', 'SL']
        with open(log_path_zs, 'r') as f:
            prev_line = ''
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if prev_line.startswith('video to text'):
                    prefix = 'V2T'
                else:
                    prev_line = line
                    continue

                metrics = line.split(' - ')
                for metric in metrics[-1:]:
                    name, value = metric.split(': ')
                    result[f'{datasets.pop(0)} {prefix}_{name}'] = value

                prev_line = line

    results[task] = result

df = pd.DataFrame.from_dict(results, orient='index')
df = df[['T2V_P@1', 'T2V_P@5', 'T2V_P@10', 'T2V_Median R', 'V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_Median R', 'AS V2T_Median R', 'AC V2T_Median R', 'SL V2T_Median R']]
# df['notes'] = notes

print(df)

df.to_csv('results_paper.csv')
    