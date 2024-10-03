import soundfile as sf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    args = parser.parse_args()

    wavs = [x.strip() for x in open(args.src, "r").readlines()]

    new_lines = ["/"]
    for wav in wavs:
        # Read the wav file
        data, sample_rate = sf.read(wav)

        # Number of samples is the length of the data array
        num_samples = len(data)

        new_lines.append(wav+"\t"+str(num_samples))
    
    with open(args.dst, "w") as f:
        f.writelines([x+"\n" for x in new_lines])
