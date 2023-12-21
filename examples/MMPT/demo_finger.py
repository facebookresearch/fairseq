import sys

import torch
import numpy as np
from pose_format import Pose

from mmpt.models import MMPTModel


def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))


pose_path = '/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/1_1_1_cam2.pose' if len(sys.argv) < 2 else sys.argv[1]
with open(pose_path, "rb") as f:
    buffer = f.read()
    pose = Pose.read(buffer)
pose = pose.normalize(pose_normalization_info(pose.header))
pose = pose.get_components(["RIGHT_HAND_LANDMARKS"])

model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/fingerclip/rwthfs_scratch_hand_dominant_aug.yaml",
    video_encoder=None,
)
model.eval()

feat = pose.body.data
pose_frames = torch.from_numpy(np.expand_dims(feat.reshape(feat.shape[0], -1), axis=0))
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
print(score_pose_and_text(pose_frames, 'Fingerspell the letter Z in German Sign Language.'))
print(score_pose_and_text(pose_frames, 'Fingerspell the letter C in German Sign Language.'))
print(score_pose_and_text(pose_frames, 'Fingerspell the letter S in German Sign Language.'))
print(score_pose_and_text(pose_frames, 'Fingerspell the letter A.'))
print(score_pose_and_text(pose_frames, 'Fingerspell the letter a in German Sign Language.'))
print(score_pose_and_text(pose_frames, 'Fingerspell the letter A in German Sign Language.'))