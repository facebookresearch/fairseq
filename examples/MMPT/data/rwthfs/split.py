import pathlib
import random

seed = 3407
raw_path = '/shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/video/'
out_path = '/shares/volk.cl.uzh/zifjia/fairseq/examples/MMPT/data/rwthfs/'

videos = list(pathlib.Path(raw_path).rglob("*.mpg"))
video_ids = list(sorted([str(v.stem) for v in videos]))

def write(filename, video_ids):
    with open(out_path + filename, 'w') as f:
        for line in video_ids:
            f.write(f"{line}\n")

write('all.txt', video_ids)

random.seed(seed)
random.shuffle(video_ids)

length = len(video_ids)
val_ratio = 0.1
val_idx = int(length * val_ratio)
test_ratio = 0.1
test_idx = val_idx + int(length * test_ratio)

video_ids_val = video_ids[:val_idx]
video_ids_test = video_ids[val_idx:test_idx]
video_ids_train = video_ids[test_idx:]

write('train.txt', video_ids_train)
write('val.txt', video_ids_val)
write('test.txt', video_ids_test)