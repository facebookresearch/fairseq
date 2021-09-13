# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import collections
import contextlib
import wave

try:
    import webrtcvad
except ImportError:
    raise ImportError("Please install py-webrtcvad: pip install webrtcvad")
import argparse
import os
import logging
from tqdm import tqdm

AUDIO_SUFFIX = '.wav'
FS_MS = 30
SCALE = 6e-5
THRESHOLD = 0.3


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        #  sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, _ in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield [b''.join([f.bytes for f in voiced_frames]),
                       voiced_frames[0].timestamp, voiced_frames[-1].timestamp]
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield [b''.join([f.bytes for f in voiced_frames]),
               voiced_frames[0].timestamp, voiced_frames[-1].timestamp]


def main(args):
    # create output folder
    try:
        cmd = f"mkdir -p {args.out_path}"
        os.system(cmd)
    except Exception:
        logging.error("Can not create output folder")
        exit(-1)

    # build vad object
    vad = webrtcvad.Vad(int(args.agg))
    # iterating over wavs in dir
    for file in tqdm(os.listdir(args.in_path)):
        if file.endswith(AUDIO_SUFFIX):
            audio_inpath = os.path.join(args.in_path, file)
            audio_outpath = os.path.join(args.out_path, file)
            audio, sample_rate = read_wave(audio_inpath)
            frames = frame_generator(FS_MS, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, FS_MS, 300, vad, frames)
            merge_segments = list()
            timestamp_start = 0.0
            timestamp_end = 0.0
            # removing start, end, and long sequences of sils
            for i, segment in enumerate(segments):
                merge_segments.append(segment[0])
                if i and timestamp_start:
                    sil_duration = segment[1] - timestamp_end
                    if sil_duration > THRESHOLD:
                        merge_segments.append(int(THRESHOLD / SCALE)*(b'\x00'))
                    else:
                        merge_segments.append(int((sil_duration / SCALE))*(b'\x00'))
                timestamp_start = segment[1]
                timestamp_end = segment[2]
            segment = b''.join(merge_segments)
            write_wave(audio_outpath, segment, sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply vad to a file of fils.')
    parser.add_argument('in_path', type=str, help='Path to the input files')
    parser.add_argument('out_path', type=str,
                        help='Path to save the processed files')
    parser.add_argument('--agg', type=int, default=3,
                        help='The level of aggressiveness of the VAD: [0-3]')
    args = parser.parse_args()

    main(args)
