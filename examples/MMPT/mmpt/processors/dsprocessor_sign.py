"""
Processors for SignCLIP
"""

import json
import os
import pickle
import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .processor import (
    MetaProcessor,
    VideoProcessor,
    TextProcessor,
)

# -------------------- SignCLIP common -----------------------

from pose_format import Pose
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.generic import pose_normalization_info
# from pose_format.utils.generic import flip_holistic, is_left_handed, pose_normalization_info

import mediapipe as mp
mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]


class PoseProcessor(VideoProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.pose_components = config.pose_components
        self.normalize_hand = config.normalize_hand
        self.flip_pose = config.flip_pose
        self.augment2d = config.augment2d
        self.augment_temporal = config.augment_temporal
        self.gaussian_noise = config.gaussian_noise
        self.max_video_len = config.max_video_len
        self.preprocess = config.preprocess
        self.anonym_pose = config.anonym_pose
        self.is_training = (config.split == 'train') and (not config.train_for_test)

        np.random.seed(42)

    def __call__(self, video_id, pose=None):
        if video_id:
            with open(os.path.join(self.vfeat_dir, video_id + ".pose"), "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

        # select components
        if self.pose_components:
            if self.pose_components == 'reduced_face':
                pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                    {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
            else:
                pose = pose.get_components(self.pose_components)
                # 3D Hand Normalization
                if self.pose_components == ['RIGHT_HAND_LANDMARKS'] and self.normalize_hand:
                    pose = self.hand_normalization(pose)

        if self.flip_pose == 'right':
            # if is_left_handed(pose):
            if np.nan_to_num(pose.get_components(["RIGHT_HAND_LANDMARKS"]).body.data).var(axis=0).sum() == 0:
                pose = flip_holistic(pose)

        if self.anonym_pose:
            from pose_anonymization.appearance import remove_appearance
            # remove appearance + add spreadthesign mean pose
            # https://github.com/sign-language-processing/pose-anonymization
            pose = remove_appearance(pose)

        if self.preprocess == 'sign-vq' or self.preprocess == 'sign-vq-original-scale':
            from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
            # reuse the preprocessing pipeline from sign-vq
            # https://github.com/sign-language-processing/sign-vq
            pose = pre_process_mediapipe(pose)

            if not self.preprocess == 'sign-vq-original-scale':
                # this removes spreadthesign mean pose
                pose = normalize_mean_std(pose)
        else:
            # normalize pose: the mean distance between the shoulders of each person equals 1
            pose = pose.normalize(self.pose_normalization_info(pose.header))
            pose = self.pose_hide_legs(pose)

        # augmentation (training only)
        if self.is_training:
            if self.flip_pose and random.random() < self.flip_pose:
                # TODO: move the flip_pose function into pose-format

                # CAUTION: flipping works on reduced set of key points only
                FLIPPED_COMPONENTS = ["POSE_LANDMARKS", "FACE_LANDMARKS", "RIGHT_HAND_LANDMARKS", "LEFT_HAND_LANDMARKS"]
                # FLIPPED_BODY_POINTS = ['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP']
                FLIPPED_BODY_POINTS = ['NOSE', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EAR', 'LEFT_EAR', 'MOUTH_RIGHT', 'MOUTH_LEFT', 'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_PINKY', 'LEFT_PINKY', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_THUMB', 'LEFT_THUMB', 'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'RIGHT_HEEL', 'LEFT_HEEL', 'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX']
                # face flipping based on https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
                FLIPPED_FACE_POINTS = ['0', '249', '10', '13', '14', '17', '251', '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300', '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362', '365', '373', '374', '375', '377', '378', '379', '152', '380', '381', '382', '384', '385', '386', '387', '388', '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466', \
                                            '7', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63', '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109', '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246']
                pose = pose.flip(0).get_components(FLIPPED_COMPONENTS, {"POSE_LANDMARKS": FLIPPED_BODY_POINTS, "FACE_LANDMARKS": FLIPPED_FACE_POINTS})
                pose = pose.normalize(self.pose_normalization_info(pose.header))
            if self.augment2d:
                pose = pose.augment2d()
            if self.augment_temporal and pose.body.data.shape[0] > 1:
                old_fps = pose.body.fps
                ratio = np.random.normal(loc=1, scale=0.2)
                if pose.body.data.shape[0] * ratio < self.max_video_len:
                    new_fps = round(old_fps * ratio)
                    pose = pose.interpolate(new_fps, kind='linear')
            if self.gaussian_noise:
                noise_std = 0.001
                noise = np.random.normal(scale=noise_std, size=pose.body.data.shape)
                pose.body.data = pose.body.data + noise

        feat = np.nan_to_num(pose.body.data)
        feat = feat.reshape(feat.shape[0], -1)
        
        return feat

    def pose_normalization_info(self, pose_header):
        if pose_header.components[0].name == "POSE_LANDMARKS":
            return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                                p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

        if pose_header.components[0].name == "BODY_135":
            return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

        if pose_header.components[0].name == "pose_keypoints_2d":
            return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))

        raise ValueError("Unknown pose header schema for normalization")

    def hand_normalization(self, pose):
        plane = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
            p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP")
        )
        line = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP")
        )
        normalizer = PoseNormalizer(plane=plane, line=line, size=100)
        tensor = normalizer(pose.body.data)

        pose.body.data = tensor
        pose.focus()

        return pose

    def pose_hide_legs(self, pose):
        if pose.header.components[0].name == "POSE_LANDMARKS":
            point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
            # pylint: disable=protected-access
            points = [
                pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                for n in point_names
                for side in ["LEFT", "RIGHT"]
            ]
            pose.body.confidence[:, :, points] = 0
            pose.body.data[:, :, points, :] = 0
            return pose
        else:
            raise ValueError("Unknown pose header schema for hiding legs")


# -------------------- RWTH Fingerspelling -----------------------


class RWTHFSMetaProcessor(MetaProcessor):
    """RWTH German Fingerspelling Database
    https://www-i6.informatik.rwth-aachen.de/aslr/fingerspelling.php
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)

        self.letter_to_id = {}
        self.id_to_letter = {}

        with open(config.gesture_id_path) as f:
            for idx, line in enumerate(f):
                letter = line.split(' = ')[1].rstrip('\n')
                self.letter_to_id[letter] = idx + 1
                self.id_to_letter[str(idx + 1)] = letter

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                signer_id, letter_id, seq_id, camera_id = video_id.split('_')

                # FIXME: for now we do full body pose estimation for all videos, so exclude cam1 where only the hands are present
                if config.video_processor == 'RWTHFSPoseProcessor' and camera_id == 'cam1':
                    continue
                
                lines.append(video_id)

            if config.split == 'train':
                self.data = []

                video_ids = defaultdict(list)
                for video_id in lines:
                    signer_id, letter_id, seq_id, camera_id = video_id.split('_')
                    video_ids[self.id_to_letter[letter_id]].append(video_id)

                length = []
                for key, value in video_ids.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in video_ids.items():
                        self.data.append(value[i % len(value)])
            else:
                self.data = lines

    def __getitem__(self, idx):
        video_id = self.data[idx]
        signer_id, letter_id, seq_id, camera_id = video_id.split('_')
        body_part = 'handshape' if camera_id == 'cam1' else 'whole body'
        text_info = f'Fingerspell the letter {self.id_to_letter[letter_id]} in German Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class RWTHFSVideoProcessor(VideoProcessor):
    def __call__(self, video_id):
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        # pooling adapater (not needed when training from scratch)
        feat_dim = 512
        if feat.shape[1] > feat_dim and not self.vfeat_custom:
            # i3d feature is 1024
            # adapt feature dimension to 512 by average pooling
            feat = feat.reshape(feat.shape[0], feat_dim, int(feat.shape[1] / feat_dim))
            feat = np.average(feat, axis=2)
        return feat


class RWTHFSPoseProcessor(PoseProcessor):
    pass

# -------------------- ASL Signs -----------------------

class ASLSignMetaProcessor(MetaProcessor):
    """Google - Isolated Sign Language Recognition
    https://www.kaggle.com/competitions/asl-signs/overview
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                lines.append(video_id)

            metadata_df = metadata_df[metadata_df['sequence_id'].isin(lines)]
            data = metadata_df.to_dict('records')

            print(f'sign distribution in the {config.split} set:')
            print(metadata_df.groupby(['sign'])['sign'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            if config.split == 'train':
                self.data = []

                indices = defaultdict(list)
                for index, item in enumerate(data):
                    indices[item['sign']].append(index)

                length = []
                for key, value in indices.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in indices.items():
                        self.data.append(data[value[i % len(value)]])
            else:
                self.data = data

    def __getitem__(self, idx):
        video_id = self.data[idx]['path'].replace('train_landmark_files/', '')
        text_info = f'Sign the sign "{self.data[idx]["sign"]}" in American Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class ASLSignPoseProcessor(PoseProcessor):
    NOSE = [
        1,2,98,327
    ]
    LIP = [ 0, 
        61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    REYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ]
    LEYE = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ]
    FACE = sorted(NOSE + LIP + REYE + LEYE)
    FACE_FULL = np.arange(0, 468).tolist()

    LHAND = np.arange(468, 489).tolist()
    POSE = np.arange(489, 522).tolist()
    RHAND = np.arange(522, 543).tolist()
    BODY = LHAND + POSE + RHAND

    def __call__(self, video_id):
        import pyarrow.parquet as pq

        pose_df = pq.read_table(os.path.join(self.vfeat_dir, video_id)).to_pandas()

        # pd.set_option('display.max_rows', None)
        # print(pose_df[pose_df['frame'] == 18])

        # pose_df = pose_df[pose_df['type'].isin(self.pose_components)]

        points = []
        if "face" in self.pose_components:
            points = points + self.FACE
        if "face_full" in self.pose_components:
            points = points + self.FACE_FULL
        if "left_hand" in self.pose_components:
            points = points + self.LHAND
        if "pose" in self.pose_components:
            points = points + self.POSE    
        if "right_hand" in self.pose_components:
            points = points + self.RHAND

        num_frames = len(pose_df['frame'].drop_duplicates())
        dimensions = ['x', 'y', 'z']

        pose_data = pose_df[dimensions].to_numpy().reshape(num_frames, -1, len(dimensions))
        pose_data = pose_data[:, points, :]

        pose_data = pose_data.reshape(num_frames, -1)
        pose_data = np.nan_to_num(pose_data)
        
        return pose_data


# -------------------- SignCLIP v1 -----------------------

import re
import string
import importlib
from tqdm import tqdm

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader


class SignCLIPMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        import tensorflow_datasets as tfds
        import sign_language_datasets.datasets
        from sign_language_datasets.datasets.config import SignDatasetConfig

        self.config = config
        self.task = config.task
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself
        self.datasets = {}
        self.data = []

        if config.test_in_vocab:
            vocab_path = './data_stat_sp_concept_dis.csv'
            vocab_df = pd.read_csv(vocab_path)
            vocab_df = vocab_df[vocab_df['count'] > 20]
            self.vocab = list(vocab_df['text'])

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name=config.config_name or 'holistic', version=version, include_video=False, include_pose="holistic", extra={'split': split_version} if split_version else {})
            splits = ['validation' if self.split == 'valid' else self.split]
            # utilize unused validation data from some datasets for pretraining as well
            if self.split == 'train' and config.use_valid_for_pretraining and dataset not in [d[0] for d in config.valid_datasets]:
                splits = ['train', 'validation'] 

            for split in splits:
                data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]

                print(f'In total {len(data_l)} raw {split} examples.')
                print('Iterate over examples ...')

                self.datasets[dataset] = {
                    'pose_header': pose_header,
                    'data_l': data_l,
                }

                count = 0
                for dataset_index, datum in enumerate(tqdm(data_l)):
                    if not (datum['pose']['data'].shape[0] > 0 and datum['pose']['data'].shape[0] <= config.max_video_len):
                        continue

                    if config.debug and count >= 1000:
                        break
                    count = count + 1

                    if self.task and self.task.startswith('identification'):
                        # for dicta_sign
                        signed_language_map = {
                            'GSL': 'gss',
                            'BSL': 'bfi',
                            'DGS': 'gsg',
                            'LSF': 'fsl',
                        }
                        signed_language = signed_language_map[datum['signed_language'].numpy().decode('utf-8')]

                        # DGS and BSL videos contain more than one camera angle, excluding for now
                        if signed_language == 'gsg' or signed_language == 'bfi':
                            continue
                        # if signed_language == 'gsg' or signed_language == 'gss':
                        #     continue
                        
                        tag_prompt = f"<en> <{signed_language}>"
                        text_prompt = f"{tag_prompt} {datum['text_en'].numpy().decode('utf-8')}" if self.task == 'identification_oracle' else tag_prompt
                    else:
                        if config.sp_universal_tagging:
                            if dataset == 'bobsl_islr':
                                tag_prompt = "<en> <bfi>" 
                            else:
                                tag_prompt = "<en> <ase>" 
                        else:
                            tag_prompt = "<American Sign Language>"

                        text_content = datum['text'].numpy().decode('utf-8')

                        if config.test_in_vocab:
                            if dataset == 'asl_citizen':
                                text_content = text_content.lower()
                                text_content = text_content.rstrip(string.digits)
                            elif dataset == 'sem_lex':
                                text_content = text_content.rstrip(string.digits)
                                text_content = text_content.rstrip('_')
                                # text_content = re.sub(r'_\d+$', '', text_content)
                                text_content = text_content.replace('_', ' ')

                            if text_content not in self.vocab:
                                continue

                        text_prompt = f"{tag_prompt} {text_content}"

                    vfeat = None
                    # save memory consumption for 3.5M BOBSL ISLR examples to under 300GB
                    # better solution: use SignCLIPMetaProcessorV2 to load data asynchronously
                    if config.pre_compute_vfeat: 
                        # reconstruct pose object
                        tf_pose = datum['pose']
                        fps = int(tf_pose["fps"].numpy())
                        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
                        pose = Pose(pose_header, pose_body)
                        vfeat = self.pose_processer(pose)

                    self.data.append(dict(
                        # datum,
                        id=f"{dataset}_{datum['id'].numpy().decode('utf-8')}",
                        text=text_prompt,
                        vfeat=vfeat,
                    ))

                # if config.debug:
                #     self.data = self.data*3000

                print(f'In total {count} wellformed {split} examples.')

        print(f'In total {len(self.data)} wellformed {split} examples from all datasets.')
        
        if self.split == 'train':
            random.shuffle(self.data)

        # Group examples by text prompts
        self.text_to_idxs = defaultdict(list)
        for idx, datum in enumerate(self.data):
            self.text_to_idxs[datum['text']].append(idx)

        print('Number of examples grouped by the text prompts:')
        text_to_idxs_num = [(text, len(idxs)) for text, idxs in self.text_to_idxs.items()]
        text_to_idxs_num = sorted(text_to_idxs_num, key=lambda x: x[1], reverse=True)
        for i, entry in enumerate(text_to_idxs_num):
            if i < 10 or (len(text_to_idxs_num) - i < 10):
                print(entry)
            elif i == 10:
                print('...')
        print('Total classes:', len(text_to_idxs_num))
        
        # unique sampler: sample config.unique_sampler_num examples of different text prompts randomly (for a batch)
        if self.split == 'train' and config.unique_sampler_num:
            if config.unique_sampler_num > len(text_to_idxs_num):
                raise ValueError(f'Impossible to sample {config.unique_sampler_num} unique examples given {len(text_to_idxs_num)} unique text prompts.')

            self.unique_sampler_num = config.unique_sampler_num
            self.text_prompts = list(self.text_to_idxs.keys())
            self.text_prompts_sampled = []

    def __getitem__(self, idx):
        if hasattr(self, 'unique_sampler_num'):
            # reset when starting a new epoch or when a batch is full
            if idx == 0 or len(self.text_prompts_sampled) == self.unique_sampler_num:
                self.text_prompts = list(self.text_to_idxs.keys())
                self.text_prompts_sampled = []
                # print('reset')

            # randomly sample one example per text prompt
            sampled_text = random.choice(self.text_prompts)
            sampled_idx = random.choice(self.text_to_idxs[sampled_text])
            self.text_prompts.remove(sampled_text)
            self.text_prompts_sampled.append(sampled_text)

            # print(sampled_idx, sampled_text)
            return sampled_idx, sampled_text
        else:
            datum = self.data[idx]

            # vfeat pre-computed during __init__
            if 'vfeat' in datum:
                return idx, datum['text'], datum['vfeat']

            # reconstruct pose object
            tf_pose = datum['pose']
            # tf_pose = dataset['data_l'][datum['dataset_index']]['pose']
            fps = int(tf_pose["fps"].numpy())
            pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
            pose = Pose(dataset['pose_header'], pose_body)
            vfeat = self.pose_processer(pose)

            return idx, datum['text'], vfeat


class SignCLIPPoseProcessor(PoseProcessor):
    def __call__(self, pose):
        # the pose objects are passed to PoseProcessor
        feat = super().__call__(None, pose)
        return feat
        

# -------------------- SignCLIP pretrained on Spreadthesign -----------------------

class SignCLIPPretrainMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        if config.debug:
            config.split = 'test'

        self.vfeat_dir = config.vfeat_dir
        self.task = config.task
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = int(line.rstrip('\n'))
                lines.append(video_id)

            metadata_df = metadata_df.reset_index()
            metadata_df = metadata_df[metadata_df['index'].isin(lines)]
            metadata_df = metadata_df[metadata_df['language'] == 'en']

            print(metadata_df)

            print(f'text distribution in the {config.split} set:')
            print(metadata_df.groupby(['text'])['text'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            print(f'language distribution in the {config.split} set:')
            print(metadata_df.groupby(['videoLanguage'])['videoLanguage'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            data = metadata_df.to_dict('records')
            data = [datum for datum in tqdm(data) if os.path.exists(os.path.join(config.vfeat_dir, datum['pose']))]

            print(f'In total {len(data)} {config.split} examples with poses.')

            self.data = data

            if config.split == 'train':
                random.shuffle(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        video_id = datum['pose'].replace('.pose', '')
        text = '' if self.task == 'identification' else datum['text']
        vlan = '<ase>' if self.task == 'conceptualization' else f"<{datum['videoLanguage']}>"
        text_info = f"<{datum['language']}> {vlan} {text}"
        return video_id, text_info


class SignCLIPMetaProcessorV2(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        import tensorflow_datasets as tfds
        import sign_language_datasets.datasets
        from sign_language_datasets.datasets.config import SignDatasetConfig

        self.config = config
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name=config.config_name or 'holistic', version=version, include_video=False, include_pose="holistic", extra={
                'split': split_version,
                'lip_feature_dir': config.lip_feature_dir if config.config_name == 'holistic_lip' else None,
            })
            split = 'validation' if self.split == 'valid' else self.split

            self.pose_header = pose_header

            if config.debug:
                split_debug = 'validation' if split == 'validation' else 'test'
                self.data_l = list(tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split_debug])[:1000]

                # Group examples by text prompts
                self.text_to_idxs = defaultdict(list)
                for idx, datum in enumerate(self.data_l):
                    self.text_to_idxs[datum['text'].numpy().decode('utf-8')].append(idx)

                print('Number of examples grouped by the text prompts:')
                text_to_idxs_num = [(text, len(idxs)) for text, idxs in self.text_to_idxs.items()]
                text_to_idxs_num = sorted(text_to_idxs_num, key=lambda x: x[1], reverse=True)
                for i, entry in enumerate(text_to_idxs_num):
                    if i < 10 or (len(text_to_idxs_num) - i < 10):
                        print(entry)
                    elif i == 10:
                        print('...')
                print('Total classes:', len(text_to_idxs_num))
            else:
                self.data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]

            print("Dataset initialized")
            print(f'In total {len(self.data_l)} raw {split} examples.')


    def reset_data_iter(self):
        print("Resetting iterator")
        self.data_iter = iter(self.data_l)


    def __len__(self): 
        return len(self.data_l)


    def __getitem__(self, idx):
        # print(f"Fetching {self.split} item {idx}")

        if idx == 0:
            self.reset_data_iter()
            self.current_idx = 0
        else:
            assert (self.current_idx + 1) == idx, "assumes sequential access items"
            self.current_idx = idx

        try:
            # Get the next data from the TensorFlow generator
            datum = next(self.data_iter)
        except StopIteration:
            print("Iterator exhausted, resetting...")
            # If the generator is exhausted, reset it
            self.reset_data_iter()
            # Retrieve the first element from the reset generator
            datum = next(self.data_iter)
            self.current_idx = 0

        example_id = datum['id'].numpy().decode('utf-8')

        tag_prompt = "<en> <bfi>" # FIXME
        text_content = datum['text'].numpy().decode('utf-8')
        text_prompt = f"{tag_prompt} {text_content}"

        # reconstruct pose object
        tf_pose = datum['pose']
        fps = int(tf_pose["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(self.pose_header, pose_body)
        vfeat = self.pose_processer(pose)

        if self.config.include_lip_reading:
            lip_feat = datum['lip'].numpy()
            vfeat = np.concatenate((vfeat, lip_feat), axis=1)

        return example_id, text_prompt, vfeat