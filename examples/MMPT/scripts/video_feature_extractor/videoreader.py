# Copyright Howto100M authors.
# Copyright (c) Facebook, Inc. All Rights Reserved

import torch as th
import pandas as pd
import os
import numpy as np
import ffmpeg
import random

from torch.utils.data import Dataset


class VideoLoader(Dataset):
    """modified from how2's video_feature_extractor."""
    def __init__(
        self,
        csv=None,
        video_dict=None,
        framerate=1,
        size=112,
        centercrop=False,
        hflip=False,
        **kwargs
    ):
        if csv is None and video_dict is None:
            raise ValueError("csv and video_dict cannot be both None.")
        if csv is not None:
            self.csv = pd.read_csv(csv)
        if video_dict is not None:
            self.csv = pd.DataFrame.from_dict(video_dict)

        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.hflip = hflip

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        return video_stream

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]
        return self._decode(output_file, video_path)

    def _decode(self, output_file, video_path):
        if not(os.path.isfile(output_file)) and os.path.isfile(video_path):
            try:
                h, w = self._get_video_dim(video_path)
            except Exception:
                print('ffprobe failed at: {}'.format(video_path))
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file}
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                height, width = self._get_output_dim(h, w)

                cmd = (
                    ffmpeg
                    .input(video_path)
                    .filter('fps', fps=self.framerate)
                    .filter('scale', width, height)
                )
                if self.hflip:
                    cmd = cmd.filter('hflip')

                if self.centercrop:
                    x = int((width - self.size) / 2.0)
                    y = int((height - self.size) / 2.0)
                    cmd = cmd.crop(x, y, self.size, self.size)
                video = self._run(cmd, output_file)
            except Exception:
                video = th.zeros(1)
        else:
            video = th.zeros(1)

        return {'video': video, 'input': video_path, 'output': output_file}

    def _run(self, cmd, output_file):
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = th.from_numpy(video.astype('float32'))
        return video.permute(0, 3, 1, 2)


class VideoVerifier(VideoLoader):
    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        try:
            return self._get_video_info(video_path)
        except Exception:
            # print('ffprobe failed at: {}'.format(video_path))
            return None


class VideoCompressor(VideoLoader):
    def __init__(
        self,
        csv=None,
        video_dict=None,
        framerate=1,
        size=112,
        centercrop=False,
        hflip=False,
        crf=32,
        **kwargs
    ):
        super().__init__(
            csv,
            video_dict,
            framerate,
            size,
            centercrop,
            hflip
        )
        self.crf = crf

    def _run(self, cmd, output_file):
        out, _ = (
            cmd.output(filename=output_file, crf=self.crf)
            .run(quiet=True)
        )
        video = None
        return video


class VideoDownloader(VideoCompressor):
    """download"""
    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]
        if not(os.path.isfile(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cmd = "wget -O" + output_file + " " + video_path
            # import subprocess
            # subprocess.check_output(
            #    cmd,
            #    stderr=subprocess.STDOUT, shell=True)
            os.system(cmd)
        return {'video': None, 'input': video_path, 'output': output_file}


class AvKeyframeVideoCompressor(VideoLoader):
    """extract keyframes from a video and save it as jpg.
    TODO: consider to merge with `CodecProcessor`.
    """
    def __init__(
        self,
        csv=None,
        video_dict=None,
        framerate=1,
        size=112,
        centercrop=False,
        max_num_frames=5,
        **kwargs
    ):
        super().__init__(csv, video_dict, framerate, size, centercrop)
        self.max_num_frames = max_num_frames

    def _get_video_dim(self, video_fn):
        """decord cannot probe the size of a video, we use pyav instead."""
        import av
        with av.open(video_fn) as container:
            height = container.streams.video[0].codec_context.height
            width = container.streams.video[0].codec_context.width
        return height, width

    def _get_output_dim(self, height, width):
        """
        keep the shorter side be `self.size`, strech the other.
        """
        if height >= width:
            return int(height * self.size / width), self.size
        else:
            return self.size, int(width * self.size / height)

    def __getitem__(self, idx):
        import av
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]
        if not(os.path.isdir(output_file)) and os.path.isfile(video_path):
            try:
                h, w = self._get_video_dim(video_path)
            except Exception:
                print('probe failed at: {}'.format(video_path))
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file}

            try:
                height, width = self._get_output_dim(h, w)

                # new for av.
                with av.open(video_path) as container:
                    container.streams.video[0].thread_type = "AUTO"
                    container.streams.video[0].codec_context.height = height
                    container.streams.video[0].codec_context.width = width
                    if self.framerate == 0:     # keyframe.
                        container.streams.video[0].codec_context.skip_frame = 'NONKEY'
                    frames = []
                    for frame in container.decode(video=0):
                        frames.append(frame)
                    frames = random.sample(frames, self.max_num_frames)

                    os.makedirs(output_file, exist_ok=True)
                    for frame in frames:
                        frame.to_image().save(
                            os.path.join(
                                output_file,
                                "%04d.jpg" % frame.index))
            except Exception:
                print('extract failed at: {}'.format(video_path))
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file}
        video = th.zeros(1)
        return {'video': video, 'input': video_path, 'output': output_file}
