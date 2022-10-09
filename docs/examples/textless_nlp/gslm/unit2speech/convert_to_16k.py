import os
import shlex
import subprocess
import progressbar
from time import time
from pathlib import Path

def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(((str(Path(f).stem)), os.path.join(root, f)))
    return out

def convert16k(inputfile, outputfile16k):
    command = ('sox -c 1 -b 16 {} -t wav {} rate 16k'.format(inputfile, outputfile16k))
    subprocess.call(shlex.split(command))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert to wav 16k audio using sox.')
    parser.add_argument('input_dir', type=str,
                    help='Path to the input dir.')
    parser.add_argument('output_dir', type=str,
                    help='Path to the output dir.')
    parser.add_argument('--extension', type=str, default='wav',
                    help='Audio file extension in the input. Default: mp3')
    args = parser.parse_args()

    # Find all sequences
    print(f"Finding all audio files with extension '{args.extension}' from {args.input_dir}...")
    audio_files = find_all_files(args.input_dir, args.extension)
    print(f"Done! Found {len(audio_files)} files.")

    # Convert to relative path
    audio_files = [os.path.relpath(file[-1], start=args.input_dir) for file in audio_files]

    # Create all the directories needed
    rel_dirs_set = set([os.path.dirname(file) for file in audio_files])
    for rel_dir in rel_dirs_set:
        Path(os.path.join(args.output_dir, rel_dir)).mkdir(parents=True, exist_ok=True)

    # Converting wavs files
    print("Converting the audio to wav files...")
    bar = progressbar.ProgressBar(maxval=len(audio_files))
    bar.start()
    start_time = time()
    for index, file in enumerate(audio_files):
        bar.update(index)
        input_file = os.path.join(args.input_dir, file)
        output_file = os.path.join(args.output_dir, os.path.splitext(file)[0]+".wav")
        convert16k(input_file, output_file)
    bar.finish()
    print(f"...done {len(audio_files)} files in {time()-start_time} seconds.")