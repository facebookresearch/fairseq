import os

import pandas as pd


def write(filename, video_ids):
    with open(filename, 'w') as f:
        for line in video_ids:
            f.write(f"{line}\n")


metadata_path = '/shares/volk.cl.uzh/amoryo/datasets/SperadTheSign.csv'
# vfeat_dir = '/shares/volk.cl.uzh/amoryo/datasets/sign-mt-poses'
# dist_dir = '/scratch/zifjia/sp_poses'

df = pd.read_csv(metadata_path)
df = df[df['language'] == 'en']
df_grouped = df.groupby(['text'])['text'].count().reset_index(name='count').sort_values(['count'], ascending=False)

print(df)

df_grouped_37 = df_grouped[df_grouped['count'] == 37]
print(df_grouped_37)
df_grouped_37 = df_grouped_37[df_grouped_37['text'].str.count(' ').eq(0)]
print(df_grouped_37)

df_filtered = df[df['text'].isin(df_grouped_37['text'])]

write('./extract_examples_iconicity.txt', df_filtered.index.values.tolist())
