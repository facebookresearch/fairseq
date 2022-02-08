import glob
import json
import argparse
from typing import final
import os
import os.path as path
import wave


EMIT_TIMESTAMP_KEY: final = "emit_after_input_word_timestamp"
COMPUTATION_DURATION_KEY: final = "computation_duration"
OUTPUT_DURATION_KEY: final = "output_play_duration"


def calc_avg_stats(log_path, gen_audio_with_discontinuity=False, sr=22_050):
    if os.path.isfile(log_path):
        log_files = [log_path]
    elif os.path.isdir(log_path):
        log_files = []
        for file in os.scandir(log_path):
            if file.path.endswith(".log"):
                log_files.append(file.path)
    else:
        log_files = glob.glob(log_path)

    if len(log_files) == 0:
        raise FileNotFoundError("Cannot find file or directory {0}".format(log_path))

    total_speaking_latency = 0
    total_compute, total_play = 0, 0
    total_discontinuity = 0

    # Loop through utterance latency logs
    for log_file in log_files:
        with open(log_file) as fp:
            data = json.load(fp)

        prev_finishing_timestamp = 0
        utt_compute_time, utt_play_time, utt_discontinuity = 0, 0, 0

        if not data:
            continue

        out_frames = []
        utt_id = path.basename(log_file).split("_")[0]
        utt_dir = os.path.dirname(log_file)
        if gen_audio_with_discontinuity:
            out_frames_discont = []

        # Loop through words
        first_word = True
        for output_word in data:
            # Start synthesizing once the text becomes available
            start_synthesis_timestamp = data[output_word][EMIT_TIMESTAMP_KEY]
            # Can only start playing if 1) the previous chunk is done playing and 2) synthesis computation has finished
            start_playing_timestamp = max(prev_finishing_timestamp,
                                          start_synthesis_timestamp + data[output_word][COMPUTATION_DURATION_KEY])

            # We have discontinuity if the previous chunk finishes playing before the next chunk is synthesized
            if 0 < prev_finishing_timestamp < start_playing_timestamp:
                discont_before_chunk = start_playing_timestamp - prev_finishing_timestamp
                utt_discontinuity += discont_before_chunk
            else:
                discont_before_chunk = 0

            # Update finishing timestamp for current chunk
            prev_finishing_timestamp = start_playing_timestamp + data[output_word][OUTPUT_DURATION_KEY]

            utt_compute_time += data[output_word][COMPUTATION_DURATION_KEY]
            utt_play_time += data[output_word][OUTPUT_DURATION_KEY]

            chunk_file = utt_dir + f"/{utt_id}_{output_word}.wav"
            if os.path.isfile(chunk_file):
                with wave.open(chunk_file, 'rb') as w:
                    # Read in frames as binary
                    cur_frames = w.readframes(w.getnframes())

                    start_idx, end_idx = trim_artifacts(cur_frames, w.getnframes())
                    chunk_frames = cur_frames[start_idx:end_idx]

                    out_frames.append([w.getparams(), cur_frames])

                    if gen_audio_with_discontinuity:
                        n_pad_frames = 0
                        if discont_before_chunk > 0:
                            n_pad_frames = int(discont_before_chunk * sr // 2) * 4
                        if first_word:   # first token, add silence when waiting
                            n_pad_frames = int(data[output_word][EMIT_TIMESTAMP_KEY] * sr // 2) * 4
                            first_word = False
                        if n_pad_frames != 0:
                            pad = bytearray([0 for _ in range(n_pad_frames)])
                            chunk_frames = pad + cur_frames[start_idx:end_idx]
                        # Take frames from the previous index onwards
                        out_frames_discont.append([w.getparams(), chunk_frames])

        # last prev_finishing_timestamp is when TTS finishes
        # last data[output_word][EMIT_TIMESTAMP_KEY] is when ground-truth input text
        speaking_latency = prev_finishing_timestamp - data[output_word][EMIT_TIMESTAMP_KEY]

        total_speaking_latency += speaking_latency
        total_compute += utt_compute_time
        total_play += utt_play_time
        total_discontinuity += utt_discontinuity

        with wave.open(path.join(utt_dir, f"{utt_id}.wav"), 'wb') as output:
            output.setparams(out_frames[0][0])
            for chunk_idx in range(len(data)):
                output.writeframes(out_frames[chunk_idx][1])

        if gen_audio_with_discontinuity:
            with wave.open(path.join(utt_dir, f"{utt_id}_discont.wav"), 'wb') as output:
                output.setparams(out_frames_discont[0][0])
                for chunk_idx in range(len(data)):
                    output.writeframes(out_frames_discont[chunk_idx][1])

    return (total_speaking_latency / len(log_files),
            total_compute / len(log_files),
            total_play / len(log_files),
            total_discontinuity / len(log_files))


def trim_artifacts(bin_stream, num_frames, threshold=256):
    """
    The outputaudio will contain undesired popping artifacts when starting or ending with large non-zero values.
    With this method, we trim the leading and trailing non-zero values exceeding a given threshold.
    """
    def is_near_zero(i):
        return abs(i) < threshold

    # Ensure we always start and end around 0
    first_near_zero_idx, last_near_zero_idx = None, None
    for i in range(num_frames):     # signed 2-byte
        val = int.from_bytes(bin_stream[2 * i:2 * i + 2], byteorder="little", signed=True)
        if is_near_zero(val):
            if first_near_zero_idx is None:
                first_near_zero_idx = 2 * i
            last_near_zero_idx = 2 * i + 2

    if first_near_zero_idx is None:
        first_near_zero_idx = 0
    if last_near_zero_idx is None:
        last_near_zero_idx = num_frames * 2
    return first_near_zero_idx, last_near_zero_idx


def main(args):
    avg_latency, avg_compute, avg_duration, avg_discontinuity = calc_avg_stats(args.log_path,
                                                                               args.generate_audio_with_discontinuity)

    with open(args.output_path, "w") as f:
        f.write(f"Avg speaking latency (s): {avg_latency:.2f}\n")
        f.write(f"Avg computation time (s): {avg_compute:.2f}\n")
        f.write(f"Avg play time (s): {avg_duration:.2f}\n")
        f.write(f"Avg discontinuity (s): {avg_discontinuity:.2f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--generate-audio-with-discontinuity", action="store_true")

    main(parser.parse_args())
