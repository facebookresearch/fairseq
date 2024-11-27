import argparse
from pathlib import Path
import torch
import numpy as np
from pose_format import Pose
import mediapipe as mp
from mmpt.models import MMPTModel


mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]
MAX_FRAMES_DEFAULT = 256  # Default truncate length, can be overridden

# Model configurations
model_configs = [
    ("default", "signclip_v1_1/baseline_temporal"),
    ("asl_citizen", "signclip_asl/asl_citizen_finetune"),
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
    
    raise ValueError(f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}. Expected one of (POSE_LANDMARKS,BODY_135,pose_keypoints_2d)")


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

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()  # .size() is torch.Size([1, frame count, 609])
    if max_frames is not None and pose_frames.size(1) > max_frames:
        print(f"pose sequnce length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating")
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


def score_pose_and_text(pose, text, max_frames, model_name="default"):
    model = models[model_name]["model"]

    pose_frames = preprocess_pose(pose, max_frames)
    caps, cmasks = preprocess_text(text)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)

    return text, float(output["score"])  # dot-product


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose and text similarity using SignCLIP.")
    parser.add_argument(
        "pose_path",
        default="/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/1_1_1_cam2.pose",
        type=Path,
        help="Path to the .pose file.",
    )
    parser.add_argument(
        "--max_frames",
        nargs="?",
        type=int,
        const=MAX_FRAMES_DEFAULT,
        default=None,
        help=f"If provided, pose sequences longer than this will be truncated, otherwise they will not. If provided without a value, will use {MAX_FRAMES_DEFAULT}, as SignCLIP can currently only support this many. If provided with a value, will use that value",
    )

    args = parser.parse_args()

    pose_path = args.pose_path
    max_frames = args.max_frames

    if not pose_path.is_file():
        print(f"Error: File {pose_path} does not exist.")
        return

    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

        print(score_pose_and_text(pose, "random text", max_frames))
        print(score_pose_and_text(pose, "house", max_frames))
        print(score_pose_and_text(pose, "<en> <ase> house", max_frames))
        print(score_pose_and_text(pose, "<en> <gsg> house", max_frames))
        print(score_pose_and_text(pose, "<en> <fsl> house", max_frames))
        print(score_pose_and_text(pose, "<en> <ase> sun", max_frames))
        print(score_pose_and_text(pose, "<en> <ase> police", max_frames))
        print(score_pose_and_text(pose, "<en> <ase> how are you?", max_frames))


if __name__ == "__main__":
    main()
