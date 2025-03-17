from tqdm import tqdm

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
import yaml


data_dir = '/scratch/shared/beegfs/zifan/tensorflow_datasets'
sd_config = SignDatasetConfig(
    name="holistic_lip", 
    include_video=False, 
    include_pose="holistic", 
    extra={
        'poses_dir': '/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe',
        'lip_feature_dir': '/scratch/shared/beegfs/zifan/bobsl/video_features/auto_asvr',
    },
)

data = tfds.load(name='bobsl_islr', builder_kwargs=dict(config=sd_config), data_dir=data_dir)

splits = ['validation', 'test']
# splits = ['validation']
# splits = ['test']
for split in splits:
    data_l = data[split]

    max_len = 128
    pose_lens = []

    print(f'Print the first 10 {split} examples:')
    for i, datum in tqdm(enumerate(data_l)):
        if i < 10:
            print(datum['id'].numpy().decode('utf-8'))
            print(datum['text'].numpy().decode('utf-8'))
            print(datum['pose']['data'].shape if 'pose' in datum else datum['pose_length']) 
            print(datum['lip'].numpy().shape) 

        pose_len = datum['pose']['data'].shape[0] if 'pose' in datum else datum['pose_length']
        pose_lens.append(pose_len)

    print(f'In total {len(pose_lens)} {split} examples.')

    pose_lens = [l for l in pose_lens if l > 0]
    print(f'{len(pose_lens)} valid poses.')
    pose_lens = [l for l in pose_lens if l <= 32]
    print(f'{len(pose_lens)} valid poses that does not exceed max length.')
    print(f'mean pose length: {sum(pose_lens) / len(pose_lens)}')

    print('=====================================')

