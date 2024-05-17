import os
import random
import importlib
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm
import numpy as np

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader

from demo_sign import embed_pose


random.seed(42)

config_file = './projects/retri/signclip_v1/baseline_asl.yaml'


def read_pose(datum, pose_header):
    # reconstruct pose object
    tf_pose = datum['pose']
    fps = int(tf_pose["fps"].numpy())
    pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
    pose = Pose(pose_header, pose_body)
    return pose

with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
        datasets = config['dataset']['train_datasets']
        data_dir = config['dataset']['data_dir']
    except yaml.YAMLError as exc:
        print(exc)

datasets = [item if len(item) == 3 else [*item, None] for item in datasets]
datasets = datasets[2:]

for dataset, version, split_version in datasets:
    print('--------------------------------')
    print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
    print('--------------------------------')

    # read common pose header for the dataset
    dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
    with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    sd_config = SignDatasetConfig(name="holistic", version=version, include_video=False, include_pose="holistic", extra={'split': split_version} if split_version else {})
    splits = ['test', 'train']

    number_per_sign = 10
    text_to_embeddings = {
        'train': {},
        'test': {},
    }

    for split in splits:
        data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=data_dir)[split]

        print(f'In total {len(data_l)} raw {split} examples.')

        # filter empty or long poses
        data_l_filtered = [datum for datum in tqdm(data_l) if datum['pose']['data'].shape[0] > 0 and datum['pose']['data'].shape[0] <= config['dataset']['max_video_len']]
        print(f'In total {len(data_l_filtered)} wellformed {split} examples.')

        embedding_directory = f'./data/{dataset}/embeddings/{split}'
        Path(embedding_directory).mkdir(parents=True, exist_ok=True)

        # Group examples by text prompts
        data_l_grouped = defaultdict(list)
        for idx, datum in tqdm(enumerate(data_l_filtered)):
            data_l_grouped[datum['text'].numpy().decode('utf-8')].append(datum)

        for text, items in tqdm(data_l_grouped.items()):
            embedding_path = f"{embedding_directory}/{text}.npy"

            if os.path.exists(embedding_path):
                print(f'{embedding_path} exists, reading ...')
                with open(embedding_path, 'rb') as f:
                    text_to_embeddings[split][text] = np.load(f)
            else:
                print(f'{embedding_path} not exists, sampling and embddding ...')

                if split == 'train':
                    data_l_grouped[text] = random.sample(items, number_per_sign)

                poses = [read_pose(datum, pose_header) for datum in data_l_grouped[text]]
                pose_embeddings = embed_pose(poses)
                text_to_embeddings[split][text] = pose_embeddings

                with open(embedding_path, 'wb') as f:
                    np.save(f, pose_embeddings)

    for text, embeddings in text_embeddings['test'].items():
        print(text)
        print(embeddings.shape)
