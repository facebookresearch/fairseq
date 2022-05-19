# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import json
import pickle
from tqdm import tqdm
import os
import numpy as np


class CaptionDedupProcessor(object):
    """remove overlapping of caption sentences(clip).
    Some statistics:
    caption:
    {'t_clip_len': 246.6448431320854,
    'video_len': 281.09174795676245,
    'clip_tps': 0.8841283727427481,
    'video_tps': 0.7821156477732097,
    'min_clip_len': 0.0,
    'max_clip_len': 398.3,
    'mean_clip_len': 3.196580003006861,
    'num_clip': 77.15897706301081}

    raw_caption:
    {'t_clip_len': 238.95908778424115,
    'video_len': 267.5914859862507,
    'clip_tps': 2.4941363624267963,
    'video_tps': 2.258989769647173,
    'min_clip_len': 0.0,
    'max_clip_len': 398.3,
    'mean_clip_len': 3.0537954186814265,
    'num_clip': 78.24986779481756}
    """

    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as fd:
            self.data = pickle.load(fd)
        self.stat = {
            "t_clip_len": [],
            "video_len": [],
            "clip_tps": [],
            "video_tps": [],
            "clip_len": [],
        }

    def __call__(self):
        for idx, video_id in enumerate(tqdm(self.data)):
            caption = json.loads(self.data[video_id])
            caption = self._dedup(caption)
            if idx < 4096:  # for the first 4096 examples, compute the statistics.
                self.save_stat(video_id, caption)
            self.data[video_id] = json.dumps(caption)
        self.print_stat()

    def single(self, video_id):
        caption = json.loads(self.data[video_id])
        for clip_idx, (start, end, text) in enumerate(
            zip(caption["start"], caption["end"], caption["text"])
        ):
            print(start, end, text)
        print("@" * 100)
        caption = self._dedup(caption)
        for clip_idx, (start, end, text) in enumerate(
            zip(caption["start"], caption["end"], caption["text"])
        ):
            print(start, end, text)
        print("#" * 100)
        self.save_stat(video_id, caption)
        self.print_stat()

    def finalize(self, tgt_fn):
        with open(tgt_fn, "wb") as fw:
            pickle.dump(self.data, fw, pickle.HIGHEST_PROTOCOL)

    def save_stat(self, video_id, caption):
        video_fn = os.path.join(
            "data/feat/feat_how2_s3d", video_id + ".npy"
        )
        if os.path.isfile(video_fn):
            with open(video_fn, "rb", 1) as fr:  # 24 is the buffer size. buffered
                version = np.lib.format.read_magic(fr)
                shape, fortran, dtype = np.lib.format._read_array_header(fr, version)
                video_len = shape[0]

            t_clip_len = 0.0
            t_tokens = 0
            for idx, (start, end, text) in enumerate(
                zip(caption["start"], caption["end"], caption["text"])
            ):
                clip_len = (
                    (end - max(caption["end"][idx - 1], start))
                    if idx > 0
                    else end - start
                )
                t_clip_len += clip_len
                t_tokens += len(text.split(" "))
                self.stat["clip_len"].append(clip_len)
            self.stat["t_clip_len"].append(t_clip_len)
            self.stat["video_len"].append(video_len)
            self.stat["clip_tps"].append(t_tokens / t_clip_len)
            self.stat["video_tps"].append(t_tokens / video_len)

    def print_stat(self):
        result = {
            "t_clip_len": np.mean(self.stat["t_clip_len"]),
            "video_len": np.mean(self.stat["video_len"]),
            "clip_tps": np.mean(self.stat["clip_tps"]),
            "video_tps": np.mean(self.stat["video_tps"]),
            "min_clip_len": min(self.stat["clip_len"]),
            "max_clip_len": max(self.stat["clip_len"]),
            "mean_clip_len": np.mean(self.stat["clip_len"]),
            "num_clip": len(self.stat["clip_len"]) / len(self.stat["video_tps"]),
        }
        print(result)

    def _dedup(self, caption):
        def random_merge(end_idx, start, end, text, starts, ends, texts):
            if random.random() > 0.5:
                # print(clip_idx, "[PARTIAL INTO PREV]", end_idx)
                # overlapped part goes to the end of previous.
                ends[-1] = max(ends[-1], start)  # ?
                rest_text = text[end_idx:].strip()
                if rest_text:
                    starts.append(max(ends[-1], start))
                    ends.append(max(end, starts[-1]))
                    texts.append(rest_text)
            else:  # goes to the beginning of the current.
                # strip the previous.
                left_text = texts[-1][:-end_idx].strip()
                if left_text:
                    # print(clip_idx, "[PREV PARTIAL INTO CUR]", end_idx)
                    ends[-1] = min(ends[-1], start)
                    texts[-1] = left_text
                else:
                    # print(clip_idx, "[PREV LEFT NOTHING ALL INTO CUR]", end_idx)
                    starts.pop(-1)
                    ends.pop(-1)
                    texts.pop(-1)
                starts.append(start)
                ends.append(end)
                texts.append(text)

        starts, ends, texts = [], [], []
        for clip_idx, (start, end, text) in enumerate(
            zip(caption["start"], caption["end"], caption["text"])
        ):
            if not isinstance(text, str):
                continue
            text = text.replace("\n", " ").strip()
            if len(text) == 0:
                continue
            starts.append(start)
            ends.append(end)
            texts.append(text)
            break

        for clip_idx, (start, end, text) in enumerate(
            zip(
                caption["start"][clip_idx + 1:],
                caption["end"][clip_idx + 1:],
                caption["text"][clip_idx + 1:],
            )
        ):
            if not isinstance(text, str):
                continue
            text = text.replace("\n", " ").strip()
            if len(text) == 0:
                continue

            # print(clip_idx, texts[-5:])
            # print(clip_idx, start, end, text)
            if texts[-1].endswith(text):  # subset of prev caption -> merge
                # print(clip_idx, "[MERGE INTO PREV]")
                ends[-1] = max(ends[-1], end)
            elif text.startswith(texts[-1]):  # superset of prev caption -> merge
                # print(clip_idx, "[PREV MERGE INTO CUR]")
                texts[-1] = text
                starts[-1] = min(starts[-1], start)
                ends[-1] = max(ends[-1], end)
            else:  # overlapping or non-overlapping.
                for end_idx in range(1, len(text) + 1):
                    if texts[-1].endswith(text[:end_idx]):
                        random_merge(end_idx, start, end, text, starts, ends, texts)
                        break
                else:
                    starts.append(start)
                    ends.append(end)
                    texts.append(text)

            assert (ends[-1] + 0.001) >= starts[-1] and len(
                texts[-1]
            ) > 0, "{} {} {} <- {} {} {}, {} {} {}".format(
                str(starts[-1]),
                str(ends[-1]),
                texts[-1],
                caption["start"][clip_idx - 1],
                caption["end"][clip_idx - 1],
                caption["text"][clip_idx - 1],
                str(start),
                str(end),
                text,
            )

        return {"start": starts, "end": ends, "text": texts}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dedup how2 caption")
    parser.add_argument('--how2dir', default="data/how2")
    args = parser.parse_args()

    raw_caption_json = os.path.join(args.how2dir, "raw_caption.json")
    raw_caption_pickle = os.path.join(args.how2dir, "raw_caption.pkl")
    raw_caption_dedup_pickle = os.path.join(args.how2dir, "raw_caption_dedup.pkl")

    def convert_to_pickle(src_fn, tgt_fn):
        with open(src_fn) as fd:
            captions = json.load(fd)

        for video_id in captions:
            captions[video_id] = json.dumps(captions[video_id])

        with open(tgt_fn, "wb") as fw:
            pickle.dump(captions, fw, pickle.HIGHEST_PROTOCOL)

    if not os.path.isfile(raw_caption_pickle):
        convert_to_pickle(raw_caption_json, raw_caption_pickle)

    deduper = CaptionDedupProcessor(raw_caption_pickle)
    deduper()
    deduper.finalize(raw_caption_dedup_pickle)

    """
    # demo
    deduper = CaptionDedupProcessor("data/how2/raw_caption.pkl")
    deduper.single("HfIeQ9pzL5U")
    """
