import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
import yaml


config_file = './projects/retri/signclip_v1/baseline_asl.yaml'

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
        datasets = config['dataset']['train_datasets']
        data_dir = config['dataset']['data_dir']
    except yaml.YAMLError as exc:
        print(exc)

datasets = [item if len(item) == 3 else [*item, None] for item in datasets]
for dataset, version, split in datasets:
    print('================================')
    print(f'Loading the {dataset} {version} dataset ... ')
    print('================================')

    extra = {'split': split} if split else {}
    if dataset == 'spread_the_sign':
        extra.update({
            'pose_dir': sp_pose_dir,
            'csv_path': sp_csv_path,
        })

    sd_config = SignDatasetConfig(name="holistic", version=version, include_video=False, include_pose="holistic", extra=extra)
    data = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=data_dir)

    splits = ['train', 'validation', 'test']
    for split in splits:
        data_l = list(data[split])
        print(f'In total {len(data_l)} {split} examples.')

        max_len = 128
        pose_lens = []

        print('Print the first 10 examples:')
        for i, datum in enumerate(data_l):
            if i < 10:
                print(datum['id'].numpy().decode('utf-8'))
                print(datum['text'].numpy().decode('utf-8'))
                print(datum['pose']['data'].shape if 'pose' in datum else datum['pose_length']) 

            pose_len = datum['pose']['data'].shape[0] if 'pose' in datum else datum['pose_length']
            pose_lens.append(pose_len)

        pose_lens = [l for l in pose_lens if l > 0]
        print(f'{len(pose_lens)} valid poses.')
        pose_lens = [l for l in pose_lens if l <= 256]
        print(f'{len(pose_lens)} valid poses that does not exceed max length.')
        print(f'mean pose length: {sum(pose_lens) / len(pose_lens)}')
