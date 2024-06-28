from collections import Counter
import importlib
import yaml
import tqdm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

from pose_format import Pose
from pose_format.pose_header import PoseHeader
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader

from demo_sign import score_pose_and_text_batch


dataset = 'dicta_sign'
data_dir = '/shares/volk.cl.uzh/zifjia/tensorflow_datasets'
sd_config = SignDatasetConfig(name="holistic", version='1.0.0', include_video=False, include_pose="holistic")
data = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=data_dir)

dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
    pose_header = PoseHeader.read(BufferReader(buffer.read()))

split = 'train'
# data_l = list(data[split])
data_l = list(data[split])[:100]

language_candidates = [
    'gss',
    # 'bfi',
    # 'gsg',
    'fsl',
]
# language_map = {
#     'gss': 'GSL',
#     'bfi': 'BSL',
#     'gsg': 'DGS',
#     'fsl': 'LSF',
# }
language_map = {
    'GSL': 'gss',
    'BSL': 'bfi',
    'DGS': 'gsg',
    'LSF': 'fsl',
}

min_len = 0
max_len = 256
pose_lens = []
c = Counter()
data = []

print('Loading data ...')
for i, datum in tqdm.tqdm(enumerate(data_l)):
    language = language_map[datum['signed_language'].numpy().decode('utf-8')]
    if language not in language_candidates:
        continue

    pose_len = datum['pose']['data'].shape[0] if 'pose' in datum else datum['pose_length']
    pose_lens.append(pose_len)

    if pose_len > min_len and pose_len <= max_len:
        c[language] += 1

        # reconstruct pose object
        tf_pose = datum['pose']
        fps = int(tf_pose["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)

        datum = {
            'id': datum['id'].numpy().decode('utf-8'),
            'language': language,
            'text': datum['text'].numpy().decode('utf-8'),
            'text_en': datum['text_en'].numpy().decode('utf-8'),
        }
        print(datum)

        datum['pose'] = pose
        data.append(datum)

pose_lens = [l for l in pose_lens if l > min_len]
print(f'{len(pose_lens)} valid poses.')
pose_lens = [l for l in pose_lens if l <= max_len]
print(f'{len(pose_lens)} valid poses that does not exceed max length.')
print(f'mean valid pose length: {sum(pose_lens) / len(pose_lens)}')

print(f'In total {len(data)} {split} examples.')
print('Count examples by languages:', c)

poses = [d['pose'] for d in data]
text_prompts = [f'<en> <{l}>' for l in language_candidates]

scores = score_pose_and_text_batch(poses, text_prompts)
print(scores)
y_pred = np.argmax(scores, axis=1)
print(y_pred)

text_labels = [d['language'] for d in data]
y_true = [language_candidates.index(t) for t in text_labels]
print(y_true)

print('Scores:')
print('Accuracy:', accuracy_score(y_true, y_pred))
print('F1 Micro averaged:', f1_score(y_true, y_pred, average='micro'))
print('F1 Separate:', f1_score(y_true, y_pred, average=None))
