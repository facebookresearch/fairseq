import argparse
from pathlib import Path
import soundfile

def get_insl_frame(parse):
    out = []
    def is_ont_token(tok):
        return tok[0] in ["[", "]"];

    res = []
    x = []
    for tok in parse.split():
        if is_ont_token(tok):
            res.extend('_'.join(x))
            x = []
            res.append(tok.upper())
        else:
            x.append(tok.upper())

    return " ".join(res) + ' | '

def sequencify_utterance(utterance):
    utterance = utterance.upper()
    utterance = utterance.replace(' ', '|') + '|'
    utterance = list(utterance)
    utterance = ' '.join(utterance)
    return utterance


def generate_fairseq_manifests(manifest, output_path, audio_root=None):

    with open(manifest, 'r') as i:
        parses = []
        utterances = []
        filepaths = []
        keys = None
        for (idx, line) in enumerate(i):
            if idx == 0: keys = line.strip().split('\t')
            else:
                data = { k: v for (k, v) in zip(keys, line.split('\t'))}
                parses.append(get_insl_frame(data['decoupled_normalized_seqlogical']))
                utterances.append(sequencify_utterance(data['normalized_utterance']))
                filepaths.append(data['file_id'])

    parses_fp = output_path.with_suffix('.parse')
    with open(str(parses_fp), 'w') as o:
        for p in parses:
            o.write(p + '\n')

    utterances_fp = output_path.with_suffix('.ltr')
    with open(str(utterances_fp), 'w') as o:
        for u in utterances:
            o.write(u + '\n')

    filepaths_fp = output_path.with_suffix('.tsv')
    with open(str(filepaths_fp), 'w') as o:
        o.write(str(audio_root) + '\n')
        for f in filepaths:
            fullpath = audio_root / f
            assert fullpath.exists(), f'{fullpath}'
            frames = soundfile.info(fullpath).frames
            o.write(f'{f}\t{frames}\n')

def main(args):

    splits = ['train', 'eval', 'test']
    root = Path(args.stop_root)
    output_root = Path(args.output)

    for split in splits:
        stop_manifest_path = root / 'manifests' / (split + '.tsv')
        output_path = output_root / (split)

        generate_fairseq_manifests(stop_manifest_path, output_path, root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--stop_root', type=str,
                    help='path to stop root directory')
    parser.add_argument('--output', type=str,
                    help='output directory')
    args = parser.parse_args()
    main(args)
