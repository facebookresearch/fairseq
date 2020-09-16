import os, time
import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch
from itertools import chain
import zipfile
import numpy as np
import tempfile
try:
    import submitit
except:
    submitit=None

def extract_feature(wav_file_path, num_mel_bins, frame_length, frame_shift, use_energy):
    sound, sample_rate = torchaudio.load_wav(wav_file_path)
    output = kaldi.fbank(
        sound,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        use_energy=use_energy
    )
    return output


def get_zipinfo(zip_file):
    with zipfile.PyZipFile(zip_file, mode='r') as zf:
        info = zf.infolist()
    feat_info_list = {}
    for i in info:
        uid = os.path.splitext(i.filename)[0]
        offset = i.header_offset + 30 + len(i.filename) # + 28
        fsize = i.file_size
        feat_info_list[uid] =  f"{zip_file}:{offset}:{fsize}"
    return feat_info_list


#def fetch_data(zip_file, info):
#    from fairseq.data.audio.feature_fetcher import fetch_features
#    data = fetch_features(info.split()[1])
#    return data

def dump_zip(zip_file, data):
    zip_file = os.path.abspath(zip_file)
    basedir=os.path.dirname(zip_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    basename=os.path.basename(zip_file)
    with tempfile.TemporaryDirectory(prefix=basename, dir=basedir) as tmpdir: 
        curdir=os.path.abspath(os.curdir)
        os.chdir(tmpdir)
        for uid, fbank_data in data:
            np.save(f"{uid}.npy", fbank_data.numpy())
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_STORED) as fzip:
            for uid, _ in data:
                fzip.write(f"{uid}.npy")
        os.chdir(curdir)

def process_features(wav_files, args):
    outputs = []
    for wf, uid in wav_files:
        outputs.append((uid, extract_feature(wf, args['num_mel_bins'], args['frame_length'], args['frame_shift'], args['use_energy'])))
    dump_zip(args['zipfile'], outputs)
    zip_info = get_zipinfo(args['zipfile'])
    feat_list = []
    for uid, fbank_data in outputs:
        feat_list.append( f"{uid}\t{zip_info[uid]}\t{fbank_data.size(0)}")
    
    ##DEBUG
    #datas = [ fetch_data(args['zipfile'], feat_list[0]), fetch_data(args['zipfile'], feat_list[1]), fetch_data(args['zipfile'], feat_list[-1]) ]
    return feat_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dirs", nargs="+", default=['-'], required=True,
                        help="input directories with audio files")
    parser.add_argument("--audio-format", choices=["flac", "wav"], default="wav")
    parser.add_argument("--use-energy", action="store_true")
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument("--frame-length", type=float, default=25.0)
    parser.add_argument("--frame-shift", type=float, default=10.0)
    parser.add_argument("--data-path", help="path to save features (with .zip)")
    parser.add_argument("--data-tsv", help="output feature tsv file")
    parser.add_argument("--parallel-num", type=int, default=1, help="parallel process to extract features")
    parser.add_argument("--parallel-logdir", default="./parallel_logdir")
    args = parser.parse_args()

    samples=[]
    for path, _, files in chain.from_iterable(os.walk(path) for path in args.audio_dirs):
        for f in files:
            if f.endswith(args.audio_format):
                if len(os.path.splitext(f)) != 2:
                    raise Exception('Expect <utt_id.extension> file name. Got: ', f)
                utt_id = os.path.splitext(f)[0]
                samples.append((os.path.join(path, f), utt_id))

    feat_list = []
    run_args = {'num_mel_bins': args.num_mel_bins, 'frame_length': args.frame_length, 'frame_shift': args.frame_shift, 'use_energy': args.use_energy}
    if args.parallel_num == 1 or submitit is None:
        run_args['zipfile']=f"{args.data_path}.zip"
        feat_list = process_features(samples, run_args)
    else:
        lsize = len(samples) // args.parallel_num + 1
        executor = submitit.AutoExecutor(folder=args.parallel_logdir)
        executor.update_parameters(timeout_min=2000, cpus_per_task=1)
        jobs = []
        for i in range(args.parallel_num): 
            run_args['zipfile']=f"{args.data_path}-{i}.zip"
            job = executor.submit(process_features, samples[lsize*i:lsize*(i+1)], run_args)
            jobs.append(job)
        is_running = True 
        while is_running:
            time.sleep(5)
            is_running = sum([job.done() for job in jobs]) < len(jobs)
        feat_list = list(chain.from_iterable([job.result() for job in jobs]))

    with open(args.data_tsv, 'w') as fp:
        fp.write("\n".join(feat_list) + "\n")
    
