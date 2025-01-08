import random
import json
import os
import pickle
import argparse
from functools import partial
from tqdm import tqdm
from pprint import pprint
from tqdm.contrib.concurrent import process_map
from collections import defaultdict

from pose_format import Pose


random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["test", "valid", "train"],
        default="test",
        type=str,
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=1, 
        help="Number of multiprocessing workers.", 
        required=False
    )
    parser.add_argument(
        "--chunksize", 
        type=int, 
        default=1, 
        help="Multiprocessing chunksize parameter.", 
        required=False
    )
    parser.add_argument(
        "--max-num", 
        type=int, 
        default=99999999999, 
        help="Maximun number of examples", 
        required=False
    )
    args = parser.parse_args()

    data = []
    vfeat_dir = '/scratch/shared/beegfs/zifan/bobsl/islr_videos' if args.split == 'test' else '/scratch/shared/beegfs/zifan/bobsl/original_videos'

    print(f'Initialize {args.split} ...')

    if args.split == 'test':
        # Read the 25K ISOLATED SIGNS annotation files from BOBSL website

        BOBSL_PATH = '/athenahomes/zifan/BOBSL/v1.4'
        annotations = {
            'test': {
                'dict': {
                    'spottings_path': f"{BOBSL_PATH}/manual_annotations/isolated_signs/verified_dict_spottings.json",
                    'range': [-3, 22],
                },
                'mouthing': {
                    'spottings_path': f"{BOBSL_PATH}/manual_annotations/isolated_signs/verified_mouthing_spottings.json",
                    'range': [-15, 4],
                },
            },
        }

        fnames = []
        labels = []
        starts = []
        ends = []

        for annotation_source, annotation in annotations[args.split].items():
            print(f'Loading {annotation_source} ...')

            file_path = annotation['spottings_path']
            if file_path.endswith('.json'):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    data = data[args.split]

                    annotation['total_num'] = sum([len(d['names']) for d in data.values()])
                    annotation['vocab'] = len(data)

                    for gloss, value in tqdm(list(data.items())):
                        for i, name in enumerate(value['names']):
                            global_time = value['global_times'][i]
                            pose_path = f"{args.split}/{annotation_source}/{gloss}/{name}-{str(global_time).replace('.', '_')}.pose"

                            fnames.append(pose_path)
                            labels.append([gloss])

        pprint(annotations[args.split])
    else:
        # Copy annotations from the vgg_islr repo

        type2offsets = {
            "prajwal_mouthing" : (-9, 11), "dict" : (-3, 22), "attention" : (-8, 18), 
            "i3d_pseudo_label" : (0, 19), "mouthing" : (-15, 4), 
            "swin_pseudo_label" : (5, 25), "other" : (-8, 8), "cos_sim" : (0, 19)
        }
        type2prob_thresh = {
            0 : # train
                {"prajwal_mouthing" : 0.8, "dict" : 0.8, "attention" : 0., "i3d_pseudo_label" : 0.5,
                "swin_pseudo_label" : 0.3, "cos_sim" : 2., "other" : 0.},

            1 : {"prajwal_mouthing" : 0.8, "dict" : 0.9, "attention" : 0., "i3d_pseudo_label" : 0.5,
            "swin_pseudo_label" : 0.3, "cos_sim" : 2., "other" : 0.},

            3 : {"prajwal_mouthing" : 0.8, "dict" : 0.8, "mouthing" : 0.8, "other" : 0.},
        }
        TRAIN_SPLIT_NUM = 0
        VAL_SPLIT_NUM = 1
        TEST_SPLIT_NUM = 3
        vocab_file = "/work/sign-language/haran/bobsl/vocab/8697_vocab.pkl"
        spotting_file = "/work/sign-language/youngjoon/islr/anno.pkl"
        fps = 25

        if args.split == "train":
            split_idx = TRAIN_SPLIT_NUM
        elif args.split == "valid":
            split_idx = VAL_SPLIT_NUM
        elif args.split == "test":
            split_idx = TEST_SPLIT_NUM

        print('Load vocab ...')
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)["words_to_id"]
            id2word = {id : word for word, id in vocab.items()}
        print('Load spotting annotations ...')
        with open(spotting_file, 'rb') as f:
            data = pickle.load(f)
        data = data["videos"]
        
        count = 0 
        unigrams = defaultdict(list)

        # print('Collect number of annotations by args.split:')
        # for index in [TRAIN_SPLIT_NUM, VAL_SPLIT_NUM, TEST_SPLIT_NUM]:
        #     print(index, len([d for d in data['args.split'] if d == index]))

        # print('Collect number of annotations by type:')
        # for anno_type, anno_thres in type2prob_thresh[0].items():
        #     print(f'{anno_type}:', len([d for d in data["anno_type"] if anno_type == d]))
        #     print(f'{anno_type} after thresholding:', len([d for i, d in enumerate(data["anno_type"]) if anno_type == d and data["mouthing_prob"][i] > anno_thres]))

        print('Load examples ...')
        anno_type = None
        for i in tqdm(range(len(data["name"]))):
            if data["split"][i] == split_idx:
                if data["word"][i] in ['_fingerspelling', '_nosigning', '_pointing', '_lexical_signing']:
                    continue

                if data["word"][i] in vocab:
                    if anno_type is None or data["anno_type"][i] == anno_type:
                        if args.split == 'test' or (data["mouthing_prob"][i] \
                            >= type2prob_thresh[split_idx][data["anno_type"][i]]):
                                pose_filename = data["name"][i].replace('.mp4', '.pose')

                                if os.path.exists(f'{vfeat_dir}/{pose_filename}'):
                                    time = int(data["mouthing_time"][i] * fps)
                                    start_offset, end_offset = type2offsets[data["anno_type"][i]]
                                    s, e = max(0, time + start_offset), time + end_offset

                                    unigrams[pose_filename].append([s, e, data["word"][i]])

                                    count = count + 1 
                                    if count == args.max_num:
                                        break

        fnames, labels, starts, ends = flatten(unigrams)

    data = []
    indices = range(len(fnames))
    process_partial = partial(process, fnames, vfeat_dir, args.split, labels, starts, ends)
    for d in process_map(process_partial, indices, max_workers=args.num_workers, chunksize=args.chunksize):
        data.append(d)

    cache_path = os.path.join('/scratch/shared/beegfs/zifan/bobsl/data_cache/', f'{args.split}.pickle')
    with open(cache_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def process(fnames, vfeat_dir, split, labels, starts, ends, idx):
    video_id = fnames[idx]
    with open(os.path.join(vfeat_dir, video_id), "rb") as f:
        buffer = f.read()
        if split == 'test':
            pose = Pose.read(buffer)
        else:
            pose = Pose.read(buffer, start_frame=starts[idx], end_frame=ends[idx])

    data = (
        idx,
        f"<bfi> {labels[idx][0]}",
        pose,
    )
    return data


def flatten(ngrams):
    unique_words = {}
    word_counter = {}
    fnames = []
    starts = []
    ends = []
    labels = []
    max_repeat = -1
    for fname in ngrams:
        for s, e, w in ngrams[fname]:
            word_counter[w] = word_counter.get(w, 0) + 1
            if max_repeat > 0 and word_counter[w] > max_repeat:
                continue

            fnames.append(fname)
            starts.append(s)
            ends.append(e)
            if isinstance(w, tuple):
                labels.append(list(w))
            else:
                labels.append([w])
                unique_words[w] = True
    # if args.split == 'train':
    #     zipped = list(zip(fnames, labels, starts, ends))
    #     random.shuffle(zipped)
    #     fnames, labels, starts, ends = zip(*zipped)
    print("Vocab size:", len(unique_words))
    return fnames, labels, starts, ends

if __name__ == "__main__":
    main()