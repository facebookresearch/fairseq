from pathlib import Path
from pose_format import Pose


def write(filename, video_ids):
    with open(out_path + filename, 'w') as f:
        for line in video_ids:
            f.write(f"{line}\n")

pose_dir = '/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/pose/'
out_path = '/shares/volk.cl.uzh/zifjia/fairseq/examples/MMPT/data/rwthfs/'
pose_paths = Path(pose_dir).rglob('*.pose')

invalid_poses = []

for i, pose_path in enumerate(pose_paths):
    buffer = open(pose_path, "rb").read()
    pose = Pose.read(buffer)
    pose = pose.get_components(["RIGHT_HAND_LANDMARKS"])

    if not pose.body.data.any():
        invalid_poses.append(pose_path.stem)

    # if i > 100:
    #     break

invalid_poses = sorted(invalid_poses, key=lambda x: int(x.split('_')[1]))
write('invalid_poses.txt', invalid_poses)