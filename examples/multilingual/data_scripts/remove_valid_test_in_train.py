import os, sys
import glob, itertools
import pandas as pd

WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)


def load_langs(path):
    with open(path) as fr:
        langs = [l.strip() for l in fr]
    return langs



def load_sentences(raw_data, split, direction):
    src, tgt = direction.split('-')
    src_path = f"{raw_data}/{split}.{direction}.{src}"
    tgt_path = f"{raw_data}/{split}.{direction}.{tgt}"
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        return [(src, open(src_path).read().splitlines()), (tgt, open(tgt_path).read().splitlines())]
    else:
        return []

def swap_direction(d):
    src, tgt = d.split('-')
    return f'{tgt}-{src}'

def get_all_test_data(raw_data, directions, split='test'):
    test_data = [ 
        x
        for dd in directions
        for d in [dd, swap_direction(dd)]
        for x in load_sentences(raw_data, split, d)
    ]
    # all_test_data = {s for _, d in test_data for s in d}
    all_test_data = {}
    for lang, d in test_data:
        for s in d:
            s = s.strip()
            lgs = all_test_data.get(s, set())
            lgs.add(lang)
            all_test_data[s] = lgs
    return all_test_data, test_data

def check_train_sentences(raw_data, direction, all_test_data, mess_up_train={}):
    src, tgt = direction.split('-')
    tgt_path = f"{raw_data}/train.{direction}.{tgt}"
    src_path = f"{raw_data}/train.{direction}.{src}"
    print(f'check training data in {raw_data}/train.{direction}')
    size = 0
    if not os.path.exists(tgt_path) or not os.path.exists(src_path):
        return mess_up_train, size
    with open(src_path) as f, open(tgt_path) as g:
        for src_line, tgt_line in zip(f, g):
            s = src_line.strip()
            t = tgt_line.strip()
            size += 1
            if s in all_test_data:
                langs = mess_up_train.get(s, set())
                langs.add(direction)
                mess_up_train[s] = langs
            if t in all_test_data:
                langs = mess_up_train.get(t, set())
                langs.add(direction)
                mess_up_train[t] = langs                
    return mess_up_train, size

def check_train_all(raw_data, directions, all_test_data):
    mess_up_train = {}
    data_sizes = {}
    for direction in directions:
        _, size = check_train_sentences(raw_data, direction, all_test_data, mess_up_train)
        data_sizes[direction] = size
    return mess_up_train, data_sizes

def count_train_in_other_set(mess_up_train):
    train_in_others  = [(direction, s) for s, directions in mess_up_train.items() for direction in directions]
    counts = {}
    for direction, s in train_in_others:
        counts[direction] = counts.get(direction, 0) + 1
    return counts

def train_size_if_remove_in_otherset(data_sizes, mess_up_train):
    counts_in_other = count_train_in_other_set(mess_up_train)
    remain_sizes = []
    for direction, count in counts_in_other.items():
        remain_sizes.append((direction, data_sizes[direction] - count, data_sizes[direction], count, 100 * count / data_sizes[direction] ))
    return remain_sizes


def remove_messed_up_sentences(raw_data, direction, mess_up_train, mess_up_train_pairs, corrected_langs):
    split = 'train'
    src_lang, tgt_lang = direction.split('-')

    tgt = f"{raw_data}/{split}.{direction}.{tgt_lang}"
    src = f"{raw_data}/{split}.{direction}.{src_lang}"
    print(f'working on {direction}: ', src, tgt)
    if not os.path.exists(tgt) or not os.path.exists(src) :
        return
    
    corrected_tgt = f"{to_folder}/{split}.{direction}.{tgt_lang}"
    corrected_src = f"{to_folder}/{split}.{direction}.{src_lang}"
    line_num = 0
    keep_num = 0
    with open(src, encoding='utf8',) as fsrc, \
        open(tgt, encoding='utf8',) as ftgt, \
        open(corrected_src, 'w', encoding='utf8') as fsrc_corrected, \
        open(corrected_tgt, 'w', encoding='utf8') as ftgt_corrected:
            for s, t in zip(fsrc, ftgt):
                s = s.strip()
                t = t.strip()
                if t not in mess_up_train \
                    and s not in mess_up_train \
                    and (s, t) not in mess_up_train_pairs \
                    and (t, s) not in mess_up_train_pairs:
                    corrected_langs.add(direction)
                    print(s, file=fsrc_corrected)
                    print(t, file=ftgt_corrected)
                    keep_num += 1
                line_num += 1
                if line_num % 1000 == 0:
                    print(f'completed {line_num} lines', end='\r')
    return line_num, keep_num

##########


def merge_valid_test_messup(mess_up_train_valid, mess_up_train_test):
    merged_mess = []
    for s in set(list(mess_up_train_valid.keys()) + list(mess_up_train_test.keys())):
        if not s:
            continue
        valid = mess_up_train_valid.get(s, set())
        test = mess_up_train_test.get(s, set())
        merged_mess.append((s, valid | test))
    return dict(merged_mess)



#########
def check_train_pairs(raw_data, direction, all_test_data, mess_up_train={}):
    src, tgt = direction.split('-')
    #a hack; TODO: check the reversed directions
    path1 = f"{raw_data}/train.{src}-{tgt}.{src}"
    path2 = f"{raw_data}/train.{src}-{tgt}.{tgt}"
    if not os.path.exists(path1) or not os.path.exists(path2) :
        return
    
    with open(path1) as f1, open(path2) as f2:
        for src_line, tgt_line in zip(f1, f2):
            s = src_line.strip()
            t = tgt_line.strip()
            if (s, t) in all_test_data or (t, s) in all_test_data:
                langs = mess_up_train.get( (s, t), set())
                langs.add(src)
                langs.add(tgt)
                mess_up_train[(s, t)] = langs
                

def load_pairs(raw_data, split, direction):
    src, tgt = direction.split('-')
    src_f = f"{raw_data}/{split}.{direction}.{src}"
    tgt_f = f"{raw_data}/{split}.{direction}.{tgt}"
    if tgt != 'en_XX':
        src_f, tgt_f = tgt_f, src_f
    if os.path.exists(src_f) and os.path.exists(tgt_f):
        return list(zip(open(src_f).read().splitlines(), 
                open(tgt_f).read().splitlines(), 
                ))
    else:
        return []

# skip_langs = ['cs_CZ', 'en_XX', 'tl_XX', 'tr_TR']
def get_messed_up_test_pairs(split, directions):
    test_pairs = [ 
        (d,  load_pairs(raw_data, split, d))
        for d in directions
    ]
    # all_test_data = {s for _, d in test_data for s in d}
    all_test_pairs = {}
    for direction, d in test_pairs:
        src, tgt = direction.split('-')
        for s in d:
            langs = all_test_pairs.get(s, set())
            langs.add(src)
            langs.add(tgt)
            all_test_pairs[s] = langs
    mess_up_train_pairs = {}                
    for direction in directions:
        check_train_pairs(raw_data, direction, all_test_pairs, mess_up_train_pairs)  
    return all_test_pairs, mess_up_train_pairs



if __name__ == "__main__":
    #######
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from-folder',  
        required=True,
        type=str)
    parser.add_argument(
        '--to-folder',  
        required=True,
        type=str)
    parser.add_argument(
        '--directions',  
        default=None,
        type=str)


    args = parser.parse_args()    
    raw_data = args.from_folder
    to_folder = args.to_folder
    os.makedirs(to_folder, exist_ok=True)

    if args.directions:
        directions = args.directions.split(',')
    else:
        raw_files = itertools.chain(
            glob.glob(f'{raw_data}/train*'),
            glob.glob(f'{raw_data}/valid*'),
            glob.glob(f'{raw_data}/test*'),
        )
        directions = [os.path.split(file_path)[-1].split('.')[1] for file_path in raw_files]
    print('working on directions: ', directions)

    ##########
    


    all_test_data, test_data = get_all_test_data(raw_data, directions, 'test')
    print('==loaded test data==')
    all_valid_data, valid_data = get_all_test_data(raw_data, directions, 'valid')
    print('==loaded valid data==')
    all_valid_test_data =  merge_valid_test_messup(all_test_data, all_valid_data)
    mess_up_train, data_sizes = check_train_all(raw_data, directions, all_valid_test_data)
    print('training messing up with valid, test data:', len(mess_up_train))
    data_situation = train_size_if_remove_in_otherset(data_sizes, mess_up_train)
    df = pd.DataFrame(data_situation, columns=['direction', 'train_size_after_remove', 'orig_size', 'num_to_remove', 'remove_percent'])
    df.sort_values('remove_percent', ascending=False)
    df.to_csv(f'{raw_data}/clean_summary.tsv', sep='\t')
    print(f'projected data clean summary in: {raw_data}/clean_summary.tsv')    

    # correct the dataset:
    all_test_pairs, mess_up_test_train_pairs = get_messed_up_test_pairs('test', directions)
    all_valid_pairs, mess_up_valid_train_pairs = get_messed_up_test_pairs('valid', directions)

    all_messed_pairs = set(mess_up_test_train_pairs.keys()).union(set(mess_up_valid_train_pairs.keys()))    
    corrected_directions = set()

    real_data_situation = []
    for direction in directions:
        org_size, new_size = remove_messed_up_sentences(raw_data, direction, mess_up_train, all_messed_pairs, corrected_directions)
        if org_size == 0:
            print(f"{direction} has size 0")
            continue
        real_data_situation.append(
            (direction, new_size, org_size, org_size - new_size, (org_size - new_size) / org_size * 100)
        )
    print('corrected directions: ', corrected_directions)
    df = pd.DataFrame(real_data_situation, columns=['direction', 'train_size_after_remove', 'orig_size', 'num_to_remove', 'remove_percent'])
    df.sort_values('remove_percent', ascending=False)
    df.to_csv(f'{raw_data}/actual_clean_summary.tsv', sep='\t')
    print(f'actual data clean summary (which can be different from the projected one because of duplications) in: {raw_data}/actual_clean_summary.tsv')        

    import shutil
    for direction in directions:
        src_lang, tgt_lang = direction.split('-')
        for split in ['train', 'valid', 'test']:
            # copying valid, test and uncorrected train
            if direction in corrected_directions and split == 'train':
                continue
            tgt = f"{raw_data}/{split}.{direction}.{tgt_lang}"
            src = f"{raw_data}/{split}.{direction}.{src_lang}"
            if not (os.path.exists(src) and os.path.exists(tgt)):
                continue
            corrected_tgt = f"{to_folder}/{split}.{direction}.{tgt_lang}"
            corrected_src = f"{to_folder}/{split}.{direction}.{src_lang}"
            print(f'copying {src} to {corrected_src}')
            shutil.copyfile(src, corrected_src)
            print(f'copying {tgt} to {corrected_tgt}')
            shutil.copyfile(tgt, corrected_tgt)   

    print('completed')