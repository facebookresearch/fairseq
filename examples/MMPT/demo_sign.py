import sys

import torch
import numpy as np
from pose_format import Pose

from mmpt.models import MMPTModel

import mediapipe as mp
mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))

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
        else:
            raise ValueError("Unknown pose header schema for hiding legs")


pose_path = '/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/1_1_1_cam2.pose' if len(sys.argv) < 2 else sys.argv[1]
with open(pose_path, "rb") as f:
    buffer = f.read()
    pose = Pose.read(buffer)
pose = pose.normalize(pose_normalization_info(pose.header))
pose = pose_hide_legs(pose)
pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                    {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})

feat = np.nan_to_num(pose.body.data)
feat = feat.reshape(feat.shape[0], -1)

model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/signclip_v1/baseline_sp_b768.yaml",
    video_encoder=None,
)
model.eval()

pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0))
# pose_frames = torch.randn(1, 30, 63)

def score_pose_and_text(pose_frames, text):
    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)
        # output = model(pose_frames, caps, cmasks, return_score=False)
        # print(output['pooled_text'].shape)
        # print(output['pooled_video'].shape)
    
    return text, float(output["score"])  # dot-product

print(score_pose_and_text(pose_frames, 'random text'))
print(score_pose_and_text(pose_frames, '<en> <ase> dream'))
print(score_pose_and_text(pose_frames, '<en> <ase> same'))
print(score_pose_and_text(pose_frames, '<en> <ase> bye'))
print(score_pose_and_text(pose_frames, '<en> <ase> TV'))