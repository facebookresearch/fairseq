import os
import json
import pickle
import argparse
from pathlib import Path
from functools import partial
from pprint import pprint
from tqdm.contrib.concurrent import process_map

import ffmpeg


BOBSL_PATH = '/athenahomes/zifan/BOBSL/v1.4'
VIDEO_DIR = '/scratch/shared/beegfs/zifan/bobsl/original_videos'
OUTPUT_DIR = '/scratch/shared/beegfs/zifan/bobsl'

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
    'train': {
        'attention': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/attention/attention_spottings.json",
            'range': [-8, 18],
        },
        'mouthing_v1': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/mouthing/mouthing_spottings_v1.json",
            'range': [-9, 11],
        },
        'mouthing_v2': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/mouthing/mouthing_spottings_v2.pkl",
            'range': [-9, 11],
        },
        'dict_v1': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/dictionary/dictionary_spottings_v1.json",
            'range': [-3, 22],
        },
        'dict_v2': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/dictionary/dictionary_spottings_v2.pkl",
            'range': [-3, 22],
        },
        'i3d_pseudo_labels': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/i3d_pseudo_labels/i3d_pseudo_labels_spottings.pkl",
            'range': [0, 19],
        },
        'exemplars': {
            'spottings_path': f"{BOBSL_PATH}/automatic_annotations/isolated_signs/exemplars/exemplar_spottings.pkl",
            'range': [0, 19],
        },
    },
}


def trim_video(input_file, output_file, start_time, end_time):
    """
    Trims a video using ffmpeg-python by specifying start and end times.
    """

    if os.path.exists(output_file):
        # print(f"Output file '{output_file}' already exists. Skipping.")
        return

    ffmpeg.input(input_file, ss=start_time, t=(end_time - start_time)) \
        .output(output_file, c="copy") \
        .global_args("-loglevel", "error") \
        .overwrite_output() \
        .run()


def process_video_by_gloss(split, annotation_source, annotation, item):
    (gloss, value) = item
    print(gloss)

    output_video_dir = f'{OUTPUT_DIR}/islr_videos/{split}/{annotation_source}/{gloss}'
    Path(output_video_dir).mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(value['names']):
        global_time = value['global_times'][i]
        prob = value['probs'][i]
        # print(name, global_time, prob)

        video_path = f'{VIDEO_DIR}/{name}.mp4'
        output_video_path = f"{output_video_dir}/{name}-{str(global_time).replace('.', '_')}.mp4"

        fps = 25
        start_time = global_time + annotation['range'][0] / fps
        end_time = global_time + annotation['range'][1] / fps

        trim_video(video_path, output_video_path, start_time, end_time)


def process_video(split, annotation_source, annotation, item):
    gloss = item['annot_word']

    output_video_dir = f'{OUTPUT_DIR}/islr_videos/{split}/{annotation_source}/{gloss}'
    Path(output_video_dir).mkdir(parents=True, exist_ok=True)

    global_time = item['annot_time']
    name = item['episode_name'].replace('.mp4', '')

    video_path = f'{VIDEO_DIR}/{name}.mp4'
    output_video_path = f"{output_video_dir}/{name}-{str(global_time).replace('.', '_')}.mp4"

    fps = 25
    start_time = global_time + annotation['range'][0] / fps
    end_time = global_time + annotation['range'][1] / fps

    trim_video(video_path, output_video_path, start_time, end_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["test", "train"],
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
        "--read-only",
        action="store_true",
    )
    args = parser.parse_args()

    split = args.split
    for annotation_source, annotation in annotations[split].items():
        print(f'Loading {annotation_source} ...')

        file_path = annotation['spottings_path']
        if file_path.endswith('.json'):
            with open(file_path, "r") as file:
                data = json.load(file)
                data = data[split]

                annotation['total_num'] = sum([len(d['names']) for d in data.values()])
                annotation['vocab'] = len(data)

                if not args.read_only:
                    func = partial(process_video_by_gloss, split, annotation_source, annotation)
                    for _ in process_map(func, data.items(), max_workers=args.num_workers):
                        pass

        elif file_path.endswith('.pkl'):
            with open(file_path, "rb") as file:

                data = pickle.load(file)
                annotation['total_num'] = len(data['episode_name'])
                annotation['vocab'] = len(set(data['annot_word']))

                if not args.read_only:
                    data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
                    
                    func = partial(process_video, split, annotation_source, annotation)
                    for _ in process_map(func, data, max_workers=args.num_workers, chunksize=1000):
                        pass

    pprint(annotations[split])

if __name__ == "__main__":
    main()
            
