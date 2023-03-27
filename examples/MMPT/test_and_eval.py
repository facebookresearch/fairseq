from pathlib import Path
import subprocess
import yaml
import pandas as pd


tasks_path = 'projects/retri/videoclip/'
tasks = [
    'test_rwthfs_zs',
    'test_rwthfs_videoclip',   
    'test_rwthfs_scratch',  
    # 'test_rwthfs_videoclip_i3d',
    'test_rwthfs_scratch_i3d',
]
notes = [
    'zero-shot VideoCLIP (S3D HowTo100M video feature)',
    'fine-tune VideoCLIP (S3D HowTo100M video feature)',
    'train from scratch (S3D HowTo100M video feature)',
    'train from scratch (I3D BSL-1K video feature)',
]

results = {}
for task in tasks:
    task_path = tasks_path + task + '.yaml'
    config = yaml.safe_load(Path(task_path).read_text())
    log_path = f"{config['eval']['save_path']}/test.log"
    command = f'python locallaunch.py {task_path} --jobtype local_predict'

    print(command)

    if not Path(log_path).is_file():
        with open(log_path, 'w') as f:
            subprocess.run(command, shell=True, stdout=f)

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

df.to_csv('results.csv')
    