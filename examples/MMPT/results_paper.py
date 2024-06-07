from pathlib import Path
import subprocess
import argparse
import yaml
import os
import re
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--overwrite', required=False, action='store_true', help='whether to overwrite existing testing logs')
args = parser.parse_args()

tasks_path = 'projects/retri/'
tasks = [
    'signclip_v1_1/baseline',
    'signclip_v1_1/baseline_layer',
    'signclip_v1_1/baseline_proj',
    'signclip_v1_1/baseline_proj_l2',
    'signclip_v1_1/baseline_reduce',
    'signclip_v1_1/baseline_pre',
    'signclip_v1_1/baseline_anonym',
    'signclip_v1_1/baseline_handedness',
    'signclip_v1_1/baseline_spatial',
    'signclip_v1_1/baseline_temporal',
    'signclip_v1_1/baseline_noise',
]
notes = [
]

datasets = ['AS', 'AC', 'SL']
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
        index = -1
        with open(log_path_zs, 'r') as f:
            prev_line = ''
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if prev_line.startswith('text to video'):
                    prefix = 'T2V'
                    index = index + 1
                elif prev_line.startswith('video to text'):
                    prefix = 'V2T'
                else:
                    prev_line = line
                    continue

                metrics = line.split(' - ')
                for metric in metrics:
                    name, value = metric.split(': ')
                    result[f'{datasets[index]} {prefix}_{name}'] = value

                prev_line = line

        log_path_train = f"{config['eval']['save_path']}/train.log"

        if os.path.exists(log_path_train):
            with open(log_path_train, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    param_regex = 'num. shared model params: (.*) \(num. trained'
                    search = re.search(param_regex, line)
                    if search:
                        param_num = search.group(1)
                        param_num_M = param_num.split(',')[0]
                        result['#params'] = param_num_M + 'M'

                    time_regex = 'done training in (.*) seconds'
                    search = re.search(time_regex, line)
                    if search:
                        seconds = float(search.group(1))
                        result['time'] = str(round(seconds / 3600)) + 'h'

    results[task] = result

df = pd.DataFrame.from_dict(results, orient='index')
metrics = ['T2V_P@1', 'T2V_P@5', 'T2V_P@10', 'T2V_Median R', 'V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_Median R']
cols = metrics.copy()
for dataset in datasets:
    cols = cols + [(dataset + ' ' + m) for m in metrics]
cols = cols + ['#params', 'time']
df = df[cols]
# df['notes'] = notes

print(df)
df.to_csv('results_paper.csv')

cols_reduced = ['V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_Median R', 'AS V2T_R@10', 'AS V2T_Median R', 'AC V2T_R@10', 'AC V2T_Median R', 'SL V2T_R@10', 'SL V2T_Median R', '#params', 'time']
df[cols_reduced].to_csv('results_paper.reduced.csv')
    