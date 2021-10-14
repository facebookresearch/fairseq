# Copyright Howto100M authors.
# Copyright (c) Facebook, Inc. All Rights Reserved

import torch as th
import torch.nn.functional as F
import math
import numpy as np
import argparse

from torch.utils.data import DataLoader
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler

from tqdm import tqdm
from pathbuilder import PathBuilder
from videoreader import VideoLoader


parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument('--vdir', type=str)
parser.add_argument('--fdir', type=str)
parser.add_argument('--hflip', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='2d',
                            help='CNN type')
parser.add_argument('--half_precision', type=int, default=0,
                            help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
parser.add_argument('--vmz_model_path', type=str, default='model/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth',
                            help='vmz model path')

args = parser.parse_args()


# TODO: refactor all args into config. (current code is from different people.)
CONFIGS = {
    "2d": {
        "fps": 1,
        "size": 224,
        "centercrop": False,
        "shards": 0,
    },
    "3d": {
        "fps": 24,
        "size": 112,
        "centercrop": True,
        "shards": 0,
    },
    "s3d": {
        "fps": 30,
        "size": 224,
        "centercrop": True,
        "shards": 0,
    },
    "vmz": {
        "fps": 24,
        "size": 112,
        "centercrop": True,
        "shards": 0,
    },
    "vae": {
        "fps": 2,
        "size": 256,
        "centercrop": True,
        "shards": 100,
    }
}

config = CONFIGS[args.type]


video_dirs = args.vdir
feature_dir = args.fdir

video_dict = PathBuilder.build(video_dirs, feature_dir, ".npy", config["shards"])

dataset = VideoLoader(
    video_dict=video_dict,
    framerate=config["fps"],
    size=config["size"],
    centercrop=config["centercrop"],
    hflip=args.hflip
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(args.type)
model = get_model(args)

with th.no_grad():
    for k, data in tqdm(enumerate(loader), total=loader.__len__(), ascii=True):
        input_file = data['input'][0]
        output_file = data['output'][0]
        if len(data['video'].shape) > 3:
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                if args.type == 'vmz':
                    n_chunk = math.ceil(n_chunk/float(3))
                    features = th.cuda.FloatTensor(n_chunk, 512).fill_(0)
                elif args.type == 's3d':
                    features = th.cuda.FloatTensor(n_chunk, 512).fill_(0)
                elif args.type == "vae":
                    features = th.cuda.LongTensor(n_chunk, 1024).fill_(0)
                else:
                    features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    factor = 1
                    if args.type == 'vmz':
                        factor = 3
                    min_ind = factor * i * args.batch_size
                    max_ind = factor * (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind:factor].cuda()
                    if args.type == '2d':
                        batch_features = model(video_batch) # (51, 487), (51, 512)
                    elif args.type == 's3d':
                        batch_features = model(video_batch)
                        batch_features = batch_features['video_embedding']
                    elif args.type == "vae":
                        # image_code.
                        batch_features = model(video_batch)
                    else:
                        batch_pred, batch_features = model(video_batch) # (51, 487), (51, 512)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[i*args.batch_size:(i+1)*args.batch_size] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    if args.type == "vae":
                        features = features.astype(np.int16)
                    else:
                        features = features.astype('float16')
                else:
                    if args.type == "vae":
                        features = features.astype(np.int32)
                    else:
                        features = features.astype('float32')
                np.save(output_file, features)
        else:
            print('Video {} error.'.format(input_file))
