import sys

import torch
import numpy as np
from pose_format import Pose
from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std

from mmpt.models import MMPTModel

import mediapipe as mp
mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]


sign_languages = [
    'swl',
    'lls',
    'dsl',
    'ise',
    'bfi',
    'gsg',
    'asq',
    'csq',
    'ssp',
    'lsl',
    'rsl',
    'eso',
    'tsm',
    'svk',
    'rsl-by',
    'psr',
    'aed',
    'cse',
    'csl',
    'icl',
    'ukl',
    'bqn',
    'ase',
    'pso',
    'fsl',
    'asf',
    'gss',
    'pks',
    'fse',
    'jsl',
    'gss-cy',
    'rms',
    'bzs',
    'csg',
    'ins',
    'mfs',
    'jos',
    'nzs',
    'ils',
    'csf',
    'ysl',
]

model_configs = [
    ('default', 'signclip_v1/baseline_sp_b768'),
    # ('asl_citizen', 'signclip_v1/baseline_sp_b768_finetune_asl_citizen'),
    # ('proj', 'signclip_v1_1/baseline_proj'),
    ('anonym', 'signclip_v1_1/baseline_anonym'),
]
models = {}

for model_name, config_path in model_configs:
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        f"projects/retri/{config_path}.yaml",
        video_encoder=None,
    )
    model.eval()
    models[model_name] = {
        'model': model,
        'tokenizer': tokenizer,
        'aligner': aligner,
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


def preprocess_pose(pose):
    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)
    pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
    
    # pose = pre_process_mediapipe(pose)
    # pose = normalize_mean_std(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()

    return pose_frames


def preprocess_text(text, model_name='default'):
    aligner = models[model_name]['aligner']
    tokenizer = models[model_name]['tokenizer']

    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    return caps, cmasks


def embed_pose(pose, model_name='default'):
    model = models[model_name]['model']

    caps, cmasks = preprocess_text('', model_name)
    poses = pose if type(pose) == list else [pose]
    embeddings = []

    for pose in poses:
        pose_frames = preprocess_pose(pose)

        with torch.no_grad():
            output = model(pose_frames, caps, cmasks, return_score=False)
            embeddings.append(output['pooled_video'].numpy())

    return np.concatenate(embeddings)


def embed_text(text, model_name='default'):
    model = models[model_name]['model']

    # pose_frames = torch.randn(1, 1, 534)
    pose_frames = torch.randn(1, 1, 609)
    texts = text if type(text) == list else [text]
    embeddings = []

    for text in texts:
        caps, cmasks = preprocess_text(text, model_name)

        with torch.no_grad():
            output = model(pose_frames, caps, cmasks, return_score=False)
            embeddings.append(output['pooled_text'].numpy())

    return np.concatenate(embeddings)


def score_pose_and_text(pose, text, model_name='default'):
    model = models[model_name]['model']

    pose_frames = preprocess_pose(pose)
    caps, cmasks = preprocess_text(text)

    with torch.no_grad():
        output = model(pose_frames, caps, cmasks, return_score=True)
    
    return text, float(output["score"])  # dot-product


def score_pose_and_text_batch(pose, text, model_name='default'):
    pose_embedding = embed_pose(pose, model_name)
    text_embedding = embed_text(text, model_name)

    scores = np.matmul(pose_embedding, text_embedding.T)
    return scores


def guess_language(pose, languages=sign_languages):
    text_prompt = "And I'm actually going to lock my wrists when I pike."
    text_prompt = "Athens"
    predictions = list(sorted([score_pose_and_text(pose, f'<en> <{lan}> {text_prompt}') for lan in languages], key=lambda t: t[1], reverse=True))
    return predictions


if __name__ == "__main__":
    pose_path = '/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/1_1_1_cam2.pose' if len(sys.argv) < 2 else sys.argv[1]

    with open(pose_path, "rb") as f:
        buffer = f.read()
        pose = Pose.read(buffer)

        print(np.matmul(embed_text('<en> <ase> house'), embed_text('<en> <ase> sun').T))
        print(np.matmul(embed_text('<en> <ase> how are you?'), embed_text('<en> <ase> sun').T))
        print(np.matmul(embed_text('<en> <gsg> sun'), embed_text('<en> <ase> sun').T))
        # print(embed_pose(pose))

        print(score_pose_and_text(pose, 'random text'))
        print(score_pose_and_text(pose, 'house'))
        print(score_pose_and_text(pose, '<en> <ase> house'))
        print(score_pose_and_text(pose, '<en> <gsg> house'))
        print(score_pose_and_text(pose, '<en> <fsl> house'))
        print(score_pose_and_text(pose, '<en> <ase> sun'))
        print(score_pose_and_text(pose, '<en> <ase> police'))
        print(score_pose_and_text(pose, '<en> <ase> how are you?'))

        # print(guess_language(pose, languages=['fsl', 'gss']))
        # print(guess_language(pose, languages=['ase', 'gsg', 'fsl', 'ise', 'bfi', 'gss']))
        # print(guess_language(pose))

        # scores = score_pose_and_text_batch([pose, pose], ['random text', '<en> <ase>'])
        # print(scores)

        # text = [
        #     '<en> <ase> Beijing',
        #     '<en> <ase> China',
        #     '<en> <ase> Tokyo',
        #     '<en> <ase> Japan',
        # ]

        # poses = [
        #     'stsddd22abeead72150e720f97b6c9f6166.pose',
        #     'stsafbea76639527924ebbff40c79520dec.pose',
        #     'stsf41d4c1f1ac7a9ae1609a7ae16045349.pose',
        #     'sts9d24faa4b6cf2a6ad9f6931485418487.pose',
        # ]

        # text_embeddings = embed_text(text)
        # print(text_embeddings)

        # pose_dir = '/home/zifjia/shares/amoryo/datasets/sign-mt-poses'
        # poses = [Pose.read(open(f'{pose_dir}/{pose}', 'rb').read()) for pose in poses]

        # pose_embeddings = embed_pose(poses)
        # print(pose_embeddings)

        # embeddings = np.vstack((text_embeddings, pose_embeddings))
        # print(embeddings)

        # with open('test.npy', 'wb') as f:
        #     np.save(f, embeddings)