import os
import pandas as pd
from pose_format import Pose
from tqdm import tqdm


sp_pose_dir = '/shares/volk.cl.uzh/amoryo/datasets/sign-mt-poses/'
sp_csv_path = '/shares/volk.cl.uzh/amoryo/datasets/SperadTheSign.csv'

df = pd.read_csv(sp_csv_path)
print(len(df))

# df = df[:10000]
df = df[df['language'] == 'en']

print(len(df))

print(f'language distribution:')
print(df.groupby(['videoLanguage'])['videoLanguage'].count().reset_index(name='count').sort_values(['count'], ascending=False))

max_len = 256
pose_lens = []

for index, row in tqdm(df.iterrows()):
    pose_path = f"{sp_pose_dir}{row['pose']}"
    if os.path.exists(pose_path):
        try:
            with open(pose_path, "rb") as f:
                pose = Pose.read(f.read())
                pose_len = pose.body.data.shape[0]
                pose_lens.append(pose_len)
        except Exception as e: 
            print(e)

print(f'{len(pose_lens)} poses.')
pose_lens = [l for l in pose_lens if l > 0]
print(f'{len(pose_lens)} valid poses.')
pose_lens = [l for l in pose_lens if l <= max_len]
print(f'{len(pose_lens)} valid poses that does not exceed max length.')
print(f'mean pose length: {sum(pose_lens) / len(pose_lens)}')