import os

import pandas as pd


# terms = [
#     'king',
#     'queen',
#     'man',
#     'woman',
# ]

# metadata_path = '/shares/volk.cl.uzh/amoryo/datasets/SperadTheSign.csv'
# vfeat_dir = '/shares/volk.cl.uzh/amoryo/datasets/sign-mt-poses'
# dist_dir = '/scratch/zifjia/sp_poses'

# df = pd.read_csv(metadata_path)

# for term in terms:
#     df_term = df[(df['text'] == term) & (df['language'] == 'en')]
#     df_term = df_term.drop_duplicates(subset=['videoLanguage'])
#     print(df_term)

#     for index, row in df_term.iterrows():
#         source_path = f"{vfeat_dir}/{row['pose']}"

#         target_filename = f"{row['text']}_{row['videoLanguage']}.pose"
#         target_path = f"{dist_dir}/{target_filename}"

#         print(f'{target_path} ---> {source_path}')
#         os.symlink(source_path, target_path)

terms = [
    'KING',
    'QUEEN',
    'MAN',
    'WOMAN1',
]

metadata_path = '/shares/volk.cl.uzh/zifjia/ASL_Citizen/splits/test.csv'
vfeat_dir = '/shares/volk.cl.uzh/zifjia/ASL_Citizen/poses'
dist_dir = '/scratch/zifjia/asl_citizen_poses'

df = pd.read_csv(metadata_path)

for term in terms:
    df_term = df[df['Gloss'] == term]
    print(df_term)

    for index, row in df_term.iterrows():
        source_path = f"{vfeat_dir}/{row['Video file'].replace('.mp4', '.pose')}"

        target_filename = row['Video file'].replace('.mp4', '.pose')
        target_path = f"{dist_dir}/{target_filename}"

        print(f'{target_path} ---> {source_path}')
        os.symlink(source_path, target_path)
