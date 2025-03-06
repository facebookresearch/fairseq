#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm  # progress bar

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

from pose_format import Pose

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from mmpt.models import MMPTModel

# Model configurations
model_configs = [
    ("default", "signclip_bsl/bobsl_islr_finetune"),
]
models = {}

for model_name, config_path in model_configs:
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        f"projects/retri/{config_path}.yaml",
        video_encoder=None,
    )
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
    }


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))
    
    raise ValueError(f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}. Expected one of (POSE_LANDMARKS, BODY_135, pose_keypoints_2d)")


def pose_hide_legs(pose):
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
    raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose, max_frames=None):
    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )

    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()  # Shape: [1, frame_count, feature_dim]
    if max_frames is not None and pose_frames.size(1) > max_frames:
        print(f"Pose sequence length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating.")
        pose_frames = pose_frames[:, :max_frames, :]

    return pose_frames


def preprocess_text(text, model_name="default"):
    aligner = models[model_name]["aligner"]
    tokenizer = models[model_name]["tokenizer"]

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name='default'):
    model = models[model_name]['model']

    caps, cmasks = preprocess_text('', model_name)
    poses = pose if type(pose) == list else [pose]
    pose_frames_l = []

    for p in poses:
        pose_frames = preprocess_pose(p)
        pose_frames_l.append(pose_frames)

    pose_frames_l = torch.cat(pose_frames_l)    
    batch_size = pose_frames_l.shape[0]

    with torch.no_grad():
        output = model(pose_frames_l, caps.repeat(batch_size, 1), cmasks.repeat(batch_size, 1), return_score=False)
        embeddings = output['pooled_video'].cpu().numpy()
    
    return embeddings


def process_batch(batch, vid, batch_index, save_dir):
    embeddings = embed_pose(batch)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract video embeddings using SignCLIP.")
    parser.add_argument(
        "--video_ids",
        type=str,
        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
        help="Path to text file containing video ids (one per line)."
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe",
        help="Directory where pose files are stored."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/sign_clip_bobsl",
        help="Directory to store SignCLIP embedding results."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for sliding window."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=32,
        help="Window size for sliding window."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing windows."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing feature files if set"
    )
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Read video ids from the provided file
    if not os.path.exists(args.video_ids):
        print(f"Error: Video IDs file not found: {args.video_ids}")
        return

    with open(args.video_ids, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    total_pose_files_found = 0

    # Process each video id
    for vid in video_ids:
        save_path = os.path.join(args.save_dir, f"{vid}.npy")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Embeddings file already exists for video {vid} at {save_path}. Skipping.")
            continue

        pose_path = os.path.join(args.pose_dir, f"{vid}.pose")
        if not os.path.exists(pose_path):
            print(f"Pose file not found for video id: {vid}")
            continue

        total_pose_files_found += 1

        # Read the entire pose file into a buffer
        with open(pose_path, "rb") as f:
            buffer = f.read()
            
        # Determine the total number of frames
        num_frames = Pose.read(buffer).body.data.shape[0]
        print(f"Video id {vid} has {num_frames} frames.")

        # Calculate total number of windows for progress bar
        total_windows = (num_frames - args.window_size) // args.stride + 1

        batch = []
        batch_index = 0
        video_embedding_batches = []  # Collect embeddings for all batches in this video

        # Iterate over sliding windows with a progress bar
        for start_frame in tqdm(range(0, num_frames - args.window_size + 1, args.stride),
                                total=total_windows,
                                desc=f"Processing video {vid}"):
            end_frame = start_frame + args.window_size
            # Extract pose window from buffer
            pose_window = Pose.read(buffer, start_frame=start_frame, end_frame=end_frame)
            batch.append(pose_window)

            # Process batch if batch size is reached
            if len(batch) == args.batch_size:
                embeddings = process_batch(batch, vid, batch_index, args.save_dir)
                video_embedding_batches.append(embeddings)
                batch_index += 1
                batch = []

        # Process any remaining poses in the last batch
        if batch:
            embeddings = process_batch(batch, vid, batch_index, args.save_dir)
            video_embedding_batches.append(embeddings)

        # Concatenate all batch embeddings along the first temporal dimension
        final_embeddings = np.concatenate(video_embedding_batches, axis=0)
        print(final_embeddings.shape)

        np.save(save_path, final_embeddings)
        print(f"Saved embeddings for video {vid} at {save_path} with shape {final_embeddings.shape}")

    print(f"Found and processed {total_pose_files_found} pose files out of {len(video_ids)} video ids.")

if __name__ == "__main__":
    main()
