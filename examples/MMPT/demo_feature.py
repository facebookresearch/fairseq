import os
import sys
import time

import numpy as np
import pandas as pd

import torch
from torchvision.io import read_video
from mmpt.models import MMPTModel


SEED = 3407
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VIDEO_PATH_SRF = '/net/cephfs/shares/easier.volk.cl.uzh/WMT_Shared_Task/srf/parallel/videos_256/'
VIDEO_PATH_FOCUSNEWS = '/net/cephfs/shares/volk.cl.uzh/EASIER/WP4/spoken-to-sign_sign-to-spoken/DSGS/FocusNews/videos/devtest/'

VIDEO_WINDOW_SIZE = 20
BATCH_SIZE = 1

def sample_frame_indices(start_idx, end_idx, clip_len=32):
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_embeddings(data, data_name, context='whole'):

    # load models
    video_model_name = "/net/cephfs/shares/volk.cl.uzh/zifjia/home/MMPT/projects/retri/videoclip/how2.yaml"
    video_model, tokenizer, aligner = MMPTModel.from_pretrained(video_model_name)
    video_model.eval()

    caps, cmasks = aligner._build_text_seq(
        tokenizer("some text", add_special_tokens=False)["input_ids"]
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    data = data.dropna()
    gb = data.groupby('filename')
    data_by_video = [gb.get_group(x) for x in gb.groups]

    video_embeddings = []

    for data_current in data_by_video:

        print(data_current)

        start = time.time()

        filename = data_current['filename'].iloc[0]
        video_path = (VIDEO_PATH_SRF + filename).replace('.srt', '.mp4')
        window_size = (VIDEO_WINDOW_SIZE / 2) if context == 'whole' else (VIDEO_WINDOW_SIZE / 4)

        for i, x in enumerate(data_current.to_dict('records')):
            if context == 'whole':
                video_middle = (x['start_original'] + x['end_original']) / 2
            elif context == 'start':
                video_middle = x['start_original']
            elif context == 'end':
                video_middle = x['end_original']
            
            video_start = video_middle - window_size
            video_end = video_middle + window_size
            video_start = max(video_start, 0)
            # video_end = min(video_end, len(videoreader))

            print(video_start, video_end)

            video_frames, _, info = read_video(video_path, video_start, video_end, pts_unit='sec')
            video_frames = video_frames.to(device)

            print(video_frames.size())

            target_fps = 30
            T, H, W, C = video_frames.size()
            T = T - T % target_fps
            video_frames = video_frames[:T].view(1, -1, target_fps, H, W, C)

            print(video_frames.size())

            end = time.time()
            print(end - start)

            with torch.no_grad():
                output = video_model(video_frames, caps, cmasks, return_score=False)
                video_features = output["pooled_video"]
                video_embeddings.append(video_features.cpu().numpy())
            
            print(video_features.size())

            end = time.time()
            print(end - start)

            exit()

    video_embeddings = np.concatenate(video_embeddings, axis=0)
    print(video_embeddings.shape)
    np.save(f'/net/cephfs/shares/volk.cl.uzh/zifjia/home/processing-shared-task-data/alignment_offset/{data_name}_vclip_{context}', video_embeddings)

if __name__ == "__main__":
    csv_url = "/net/cephfs/shares/volk.cl.uzh/zifjia/home/processing-shared-task-data/alignment_offset/SRF_segmented_Surrey.csv"
    data_srf = pd.read_csv(csv_url)
    print(data_srf)

    get_embeddings(data=data_srf, data_name="srf", context="whole")
    # get_embeddings(data=data_srf, data_name="srf", context="start")
    # get_embeddings(data=data_srf, data_name="srf", context="end")
