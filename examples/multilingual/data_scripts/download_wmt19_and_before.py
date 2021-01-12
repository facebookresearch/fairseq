from typing import NamedTuple, List
from urllib.parse import urlparse
import os, sys
import subprocess
from subprocess import check_call, check_output
import glob
import wget
import re
import multiprocessing as mp
from functools import partial
import pathlib
from collections import OrderedDict 

WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)

# scripts and data locations
CWD = os.getcwd()
UTILS = f"{CWD}/utils"

MOSES = f"{UTILS}/mosesdecoder"
SGM_TOOL = f'{MOSES}/scripts/ems/support/input-from-sgm.perl'

TMX2CORPUS = f"{UTILS}/tmx2corpus"
TMX_TOOL = f'python {TMX2CORPUS}/tmx2corpus.py'

to_data_path = f'{WORKDIR_ROOT}/wmt'
download_to = f'{to_data_path}/downloads'
manually_downloads = f'{to_data_path}/downloads'
extract_to = f'{to_data_path}/extracted'
#DESTDIR=${WORKDIR_ROOT}/ML50/raw/
raw_data = f'{WORKDIR_ROOT}/ML50/raw'
####

class DLDataset(NamedTuple):
    name: str
    train_urls: List[str]
    valid_urls: List[str]
    test_urls: List[str]        
    train_files_patterns: List[str] = []
    valid_files_patterns: List[str] = []
    test_files_patterns: List[str] = []



def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] Ks" % (current / total * 100, current / 1000, total / 1000), end='\r')

def get_downloaded_file(dl_folder, url):
    if isinstance(url, tuple):
        url, f = url
    else:
        url_f = urlparse(url)
        # f = os.path.split(url_f.path)[-1]
        f = '_'.join(url_f.path.split('/')[1:])
    return url, f"{dl_folder}/{f}"

def download_parts_and_combine(dl_folder, urls, filename):
    parts = []
    for url_record in urls:
        url, part_file = get_downloaded_file(dl_folder, url_record)     
        if os.path.exists(part_file):
            print(f'{part_file} has already been downloaded so skip')
        else: 
            part_file = wget.download(url, part_file, bar=bar_custom)  
        parts.append(part_file)

    def get_combine_cmd(parts):           
        #default as tar.gz.??
        return f'cat {" ".join(parts)} > {filename}'

    combine_cmd = get_combine_cmd(parts)
    call(combine_cmd, debug=True)
    return filename

def download_a_url(dl_folder, url):
    url, filename = get_downloaded_file(dl_folder, url)        
    if os.path.exists(filename):
        print(f'{filename} has already been downloaded so skip')
        return filename

    print(f'downloading {url} to {filename}')
    if isinstance(url, list) or isinstance(url, tuple):
        download_parts_and_combine(dl_folder, url, filename)
    else:
        wget.download(url, filename, bar=bar_custom)
    print(f'dowloaded: {filename}')
    return filename

def download_files(dl_folder, urls, completed_urls={}):
    for url_record in urls:
        url, _ = get_downloaded_file(dl_folder, url_record) 
        filename = download_a_url(dl_folder, url_record) 
        completed_urls[str(url)] = filename
    return completed_urls

def check_need_manual_downalod(dl_folder, to_manually_download_urls):
    to_be_manually_dowloaded = []
    manually_completed_urls = {}
    for url_record, instruction in to_manually_download_urls:
        url, filename = get_downloaded_file(dl_folder, url_record)
        if not os.path.exists(filename):
            print(f'{url} need to be download manually, please download it manually following {instruction}; and copy it to {filename}')
            to_be_manually_dowloaded.append((url, filename))
        else:
            manually_completed_urls[url] = filename
    # if len(to_be_manually_dowloaded) > 0:
    #     raise ValueError('Missing files that need to be downloaded manually; stop the process now.')
    return to_be_manually_dowloaded
        
def download_dataset(to_folder, dl_dataset, completed_urls={}):
    download_files(to_folder, dl_dataset.train_urls, completed_urls)
    download_files(to_folder, dl_dataset.valid_urls, completed_urls)
    download_files(to_folder, dl_dataset.test_urls, completed_urls)
    print('completed downloading')
    return completed_urls

def call(cmd, debug=False):
    if debug:
        print(cmd)
    check_call(cmd, shell=True)

    
def get_extract_name(file_path):
    path = os.path.split(file_path)
    return path[-1] + '_extract' #.split('.')[0]

def extract_file(downloaded_file, extract_folder, get_extract_name=get_extract_name, debug=False):
    extract_name = get_extract_name(downloaded_file)
    extract_to = f'{extract_folder}/{extract_name}'
    os.makedirs(extract_to, exist_ok=True)
    if os.path.exists(f'{extract_to}/DONE'):
        print(f'{downloaded_file} has already been extracted to {extract_to} so skip')
        return extract_to
    def get_extract_cmd(filename):
        if filename.endswith('.tgz') or filename.endswith('tar.gz'):
            return f'tar xzfv {filename} -C {extract_to}'
        elif filename.endswith('.gz.tar'): 
            return f'tar xfv {filename} -C {extract_to}; (cd {extract_to}; gzip -d *.gz; [ $? -eq 0 ]  || gzip -d */*.gz)'  
        elif filename.endswith('.tar'):
            return f'tar xfv {filename} -C {extract_to}'        
        elif filename.endswith('.gz'):
            return f'cp {filename} {extract_to}; (cd {extract_to}; gzip -d *.gz)'
        elif filename.endswith('.zip'):
            return f'unzip {filename} -d {extract_to}'        
    extract_cmd = get_extract_cmd(downloaded_file) 
    print(f'extracting {downloaded_file}')
    if isinstance(extract_cmd, list):
        for c in  extract_cmd:
            call(c, debug=debug)
    else:
        call(extract_cmd, debug=debug)
    call(f'echo DONE > {extract_to}/DONE')
    return extract_to


def extract_all_files(
    completed_urls, extract_folder,
    get_extract_name=get_extract_name,
    completed_extraction={},
    debug=False):
    extracted_folders = OrderedDict()
    for url, downloaded_file in set(completed_urls.items()):
        if downloaded_file in completed_extraction:
            print(f'{downloaded_file} is already extracted; so skip')
            continue
        folder = extract_file(downloaded_file, extract_folder, get_extract_name, debug)
        extracted_folders[url] = folder
    return extracted_folders


def my_glob(folder):
    for p in [f'{folder}/*', f'{folder}/*/*', f'{folder}/*/*/*']:
        for f in glob.glob(p):
            yield f


def sgm2raw(sgm, debug):
    to_file = sgm[0:len(sgm) - len('.sgm')]
    if os.path.exists(to_file):
        debug and print(f'{sgm} already converted to {to_file}; so skip')
        return to_file
    cmd = f'{SGM_TOOL} < {sgm} > {to_file}'
    call(cmd, debug)
    return to_file

def tmx2raw(tmx, debug):
    to_file = tmx[0:len(tmx) - len('.tmx')]
    to_folder = os.path.join(*os.path.split(tmx)[:-1])
    if os.path.exists(f'{to_folder}/bitext.en'):
        debug and print(f'{tmx} already extracted to {to_file}; so skip')
        return to_file
    cmd = f'(cd {to_folder}; {TMX_TOOL} {tmx})'
    call(cmd, debug)
    return to_file

CZENG16_REGEX = re.compile(r'.*?data.plaintext-format/0[0-9]train$')
WMT19_WIKITITLES_REGEX = re.compile(r'.*?wikititles-v1.(\w\w)-en.tsv.gz')
TSV_REGEX = re.compile(r'.*?(\w\w)-(\w\w).tsv$')



def cut_wikitles(wiki_file, debug):
    # different languages have different file names: 
    if wiki_file.endswith('wiki/fi-en/titles.fi-en'):
        to_file1 = f'{wiki_file}.fi'
        to_file2 = f'{wiki_file}.en'
        BACKSLASH = '\\'
        cmd1 = f"cat {wiki_file} | sed 's/|||/{BACKSLASH}t/g' |cut -f1 |awk '{{$1=$1}};1' > {to_file1}"
        cmd2 = f"cat {wiki_file} | sed 's/|||/{BACKSLASH}t/g' |cut -f2 |awk '{{$1=$1}};1' > {to_file2}"  
#     elif WMT19_WIKITITLES_REGEX.match(wiki_file):
#         src = WMT19_WIKITITLES_REGEX.match(wiki_file).groups()[0]
#         to_file1 = f'{wiki_file}.{src}'
#         to_file2 = f'{wiki_file}.en'
#         cmd1 = f"cat {wiki_file} | cut -f1 |awk '{{$1=$1}};1' > {to_file1}"
#         cmd2 = f"cat {wiki_file} | cut -f2 |awk '{{$1=$1}};1' > {to_file2}"
    else:
        return None
    if os.path.exists(to_file1) and os.path.exists(to_file2):
        debug and print(f'{wiki_file} already processed to {to_file1} and {to_file2}; so skip')
        return wiki_file    

    call(cmd1, debug=debug)
    call(cmd2, debug=debug)
    return wiki_file

def cut_tsv(file, debug):
    m = TSV_REGEX.match(file)
    if m is None:
        raise ValueError(f'{file} is not matching tsv pattern')
    src = m.groups()[0]
    tgt = m.groups()[1]

    to_file1 = f'{file}.{src}'
    to_file2 = f'{file}.{tgt}' 
    cmd1 = f"cat {file} | cut -f1 |awk '{{$1=$1}};1' > {to_file1}"
    cmd2 = f"cat {file} | cut -f2 |awk '{{$1=$1}};1' > {to_file2}"         
    if os.path.exists(to_file1) and os.path.exists(to_file2):
        debug and print(f'{file} already processed to {to_file1} and {to_file2}; so skip')
        return file    

    call(cmd1, debug=debug)
    call(cmd2, debug=debug)
    return file    

    
def convert_file_if_needed(file, debug):
    if file.endswith('.sgm'):
        return sgm2raw(file, debug)
    elif file.endswith('.tmx'):
        return tmx2raw(file, debug)
    elif file.endswith('wiki/fi-en/titles.fi-en'):
        return cut_wikitles(file, debug)
#     elif WMT19_WIKITITLES_REGEX.match(file):
#         return cut_wikitles(file, debug)
    elif file.endswith('.tsv'):
        return cut_tsv(file, debug)
    elif CZENG16_REGEX.match(file):
        return convert2czeng17(file, debug)
    else:
        return file


def convert_files_if_needed(extracted_foldrs, my_glob=my_glob, debug=False):
    return {
        url: list(sorted(set(convert_file_if_needed(f, debug)) for f in sorted(set(my_glob(folder)))))
        for url, folder in extracted_foldrs.items()
    }
        
def match_patt(file_path, file_pattern, src, tgt, lang):    
    return file_pattern.format(src=src, tgt=tgt, lang=lang) in file_path

def match_patts(file_path, file_patterns, src, tgt, lang):
    for file_pattern in file_patterns:
        params = { k: v for k, v in [('src', src), ('tgt', tgt), ('lang', lang)] if k in file_pattern}
        matching = file_pattern.format(**params)   

        if isinstance(file_pattern, tuple):
            pattern, directions = file_pattern
            if f'{src}-{tgt}' in directions and matching in file_path:
                return True
        else:
            if matching in file_path:
                return True
    return False

def extracted_glob(extracted_folder, file_patterns, src, tgt, lang):
    def get_matching_pattern(file_pattern):
        params = {
            k: v 
            for k, v in [('src', src), ('tgt', tgt), ('lang', lang)] 
            if '{' + k + '}' in file_pattern
        }
        file_pattern = re.sub(r'{src:(.*?)}', r'\1' if lang == src else '', file_pattern)
        file_pattern = re.sub(r'{tgt:(.*?)}', r'\1' if lang == tgt else '', file_pattern)
        file_pattern = file_pattern.format(**params)
        return file_pattern
    for file_pattern in file_patterns:
        if isinstance(file_pattern, tuple):
            file_pattern, lang_pairs = file_pattern
            if f'{src}-{tgt}' not in lang_pairs:
                continue
#         print('working on pattern: ', file_pattern, lang_pairs )
        matching_pattern = get_matching_pattern(file_pattern)
        if matching_pattern is None:
            continue
        glob_patterns = f'{extracted_folder}/{matching_pattern}'
#         print('glob_patterns: ', glob_patterns)
        for f in glob.glob(glob_patterns):
            yield f       

# for debug usage
def all_extracted_files(split, src, tgt, extracted_folders, split_urls):
    def get_url(url):
        if isinstance(url, tuple):
            url, downloaded_file = url        
        return url
    return [
        f
        for url in split_urls
        for f in my_glob(extracted_folders[str(get_url(url))])        
    ]

def concat_files(split, src, tgt, extracted_folders, split_urls, path_patterns, to_folder, debug=False):
#     if debug:
#         print('extracted files to be filtered by patterns: ', 
#               '\n\t'.join(sorted(all_extracted_files(split, src, tgt, extracted_folders, split_urls))))
    for lang in [src, tgt]:
        to_file = f'{to_folder}/{split}.{src}-{tgt}.{lang}'
        s_src, s_tgt, s_lang = src.split('_')[0], tgt.split('_')[0], lang.split('_')[0]
        files = []
        for url in split_urls:
            if isinstance(url, tuple):
                url, downloaded_file = url
            if str(url) not in extracted_folders:
                print(f'warning: {url} not in extracted files')
            for extracted_file in set(
                extracted_glob(
                    extracted_folders[str(url)], path_patterns, 
                    s_src, s_tgt, s_lang)):
                files.append(extracted_file)
        if len(files) == 0:
            print('warning: ', f'No files found for split {to_file}')
            continue
        files = sorted(set(files))
        print(f'concating {len(files)} files into {to_file}')
        cmd = ['cat'] + [f'"{f}"' for f in files] + [f'>{to_file}']
        cmd = " ".join(cmd)
        call(cmd, debug=debug)

UTILS = os.path.join(pathlib.Path(__file__).parent, 'utils')
LID_MODEL = f'{download_to}/lid.176.bin'
LID_MULTI = f'{UTILS}/fasttext_multi_filter.py'

def lid_filter(split, src, tgt, from_folder, to_folder, debug=False):
    if not os.path.exists(LID_MODEL):
        call(f'wget -nc https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O {LID_MODEL}')
    from_prefix = f'{from_folder}/{split}.{src}-{tgt}'
    to_prefix = f'{to_folder}/{split}.{src}-{tgt}'
    if os.path.exists(f'{from_prefix}.{src}') and os.path.exists(f'{from_prefix}.{tgt}'):
        s_src, s_tgt = src.split('_')[0], tgt.split('_')[0]  
        cmd = (
            f'python {LID_MULTI} --model {LID_MODEL} --inputs {from_prefix}.{src} {from_prefix}.{tgt} '
            f'--langs {s_src} {s_tgt} --outputs {to_prefix}.{src} {to_prefix}.{tgt}'
        )
        print(f'filtering {from_prefix}')
        call(cmd, debug=debug)

def concat_into_splits(dl_dataset, src, tgt, extracted_folders, to_folder, debug):
    to_folder_tmp = f"{to_folder}_tmp"
    os.makedirs(to_folder_tmp, exist_ok=True)
    concat_files('train', src, tgt,
                 extracted_folders,
                 split_urls=dl_dataset.train_urls,
                 path_patterns=dl_dataset.train_files_patterns,
                 to_folder=to_folder_tmp, debug=debug)
    lid_filter('train', src, tgt, to_folder_tmp, to_folder, debug)

    concat_files('valid', src, tgt,
                 extracted_folders, 
                 split_urls=dl_dataset.valid_urls, 
                 path_patterns=dl_dataset.valid_files_patterns, 
                 to_folder=to_folder, debug=debug)
    concat_files('test', src, tgt, 
                 extracted_folders, 
                 split_urls=dl_dataset.test_urls, 
                 path_patterns=dl_dataset.test_files_patterns, 
                 to_folder=to_folder, debug=debug)
            

def download_multi(dl_folder, extract_folder, urls, num_processes=8, debug=False):
    pool = mp.Pool(processes=num_processes)
    download_f = partial(download_a_url, dl_folder)
    downloaded_files = pool.imap_unordered(download_f, urls)
    pool.close()
    pool.join()

BLEU_REGEX = re.compile("^BLEU\\S* = (\\S+) ")
def run_eval_bleu(cmd):
    output = check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
    print(output)
    bleu = -1.0
    for line in output.strip().split('\n'):
        m = BLEU_REGEX.search(line)
        if m is not None:
            bleu = m.groups()[0]
            bleu = float(bleu)
            break
    return bleu

def check_wmt_test_bleu(raw_folder, wmt_lang_pairs):
    not_matchings = []
    for wmt, src_tgts in wmt_lang_pairs:
        for src_tgt in src_tgts:
            print(f'checking test bleus for: {src_tgt} at {wmt}')
            src, tgt = src_tgt.split('-')
            ssrc, stgt = src[:2], tgt[:2]
            if os.path.exists(f'{raw_folder}/test.{tgt}-{src}.{src}'):
                # reversed direction may have different test set
                test_src = f'{raw_folder}/test.{tgt}-{src}.{src}'
            else:
                test_src = f'{raw_folder}/test.{src}-{tgt}.{src}'
            cmd1 = f'cat {test_src} | sacrebleu -t "{wmt}" -l {stgt}-{ssrc}; [ $? -eq 0 ] || echo ""'
            test_tgt = f'{raw_folder}/test.{src}-{tgt}.{tgt}'       
            cmd2 = f'cat {test_tgt} | sacrebleu -t "{wmt}" -l {ssrc}-{stgt}; [ $? -eq 0 ] || echo ""'
            bleu1 = run_eval_bleu(cmd1) 
            if bleu1 != 100.0:
                not_matchings.append(f'{wmt}:{src_tgt} source side not matching: {test_src}')
            bleu2 = run_eval_bleu(cmd2) 
            if bleu2 != 100.0:
                not_matchings.append(f'{wmt}:{src_tgt} target side not matching: {test_tgt}')
    return not_matchings         
 
def download_and_extract(
    to_folder, lang_pairs, dl_dataset, 
    to_manually_download_urls, 
    completed_urls={}, completed_extraction={},
    debug=False):

    dl_folder = f'{to_folder}/downloads'
    extract_folder = f'{to_folder}/extracted'
    raw_folder =  f'{to_folder}/raw'
    lid_filtered = f'{to_folder}/lid_filtered'

    os.makedirs(extract_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(lid_filtered, exist_ok=True)

    
    to_be_manually_dowloaded = check_need_manual_downalod(dl_folder, to_manually_download_urls)

    completed_urls = download_dataset(
        dl_folder, dl_dataset, completed_urls)
    if debug:
        print('completed urls: ', completed_urls)
    

    extracted_folders = extract_all_files(
        completed_urls,
        extract_folder=extract_folder, 
        completed_extraction=completed_extraction,
        debug=debug)
    if debug:
        print('download files have been extracted to folders: ', extracted_folders)

    converted_files = convert_files_if_needed(extracted_folders, debug=False)
    for src_tgt in lang_pairs:
        print(f'working on {dl_dataset.name}: {src_tgt}')
        src, tgt = src_tgt.split('-')
        concat_into_splits(dl_dataset, 
                            src=src, tgt=tgt,
                            extracted_folders=extracted_folders, 
                            to_folder=raw_folder, debug=debug)                            
    print('completed data into: ', raw_folder)

def download_czang16(download_to, username=None):
    wgets = [
        f'wget --user={username} --password=czeng -P {download_to} http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.{i}.tar'
        for i in range(10)]
    cmds = []
    for i, cmd in enumerate(wgets):
        filename = f'{download_to}/data-plaintext-format.{i}.tar'
        if os.path.exists(filename):
            print(f'{filename} has already been downloaded; so skip')
            continue
        cmds.append(cmd)
    if cmds and username is None:
        raise ValueError('No czeng username is given; please register at http://ufal.mff.cuni.cz/czeng/czeng16 to obtain username to download')        
    for cmd in cmds:
        call(cmd)
    print('done with downloading czeng1.6')

def download_czeng17_script(download_to, extract_folder, debug=False):
    url = 'http://ufal.mff.cuni.cz/czeng/download.php?f=convert_czeng16_to_17.pl.zip'
    filename = f'{download_to}/convert_czeng16_to_17.pl.zip'
    extract_to = f'{extract_folder}/{get_extract_name(filename)}'
    script_path = f'{extract_to}/convert_czeng16_to_17.pl'
    
    if not os.path.exists(script_path):
        wget.download(url, filename, bar=bar_custom)  
        extract_to = extract_file(f'{download_to}/convert_czeng16_to_17.pl.zip', extract_folder, get_extract_name=get_extract_name, debug=debug)    
    return script_path

czeng17_script_path = ""
def convert2czeng17(file, debug):
    en_file = f'{file}.en'
    cs_file = f'{file}.cs'
    
    if not os.path.exists(en_file) or not os.path.exists(cs_file):
        cs_cmd = f'cat {file} | perl {czeng17_script_path} | cut -f3 > {cs_file}'
        en_cmd = f'cat {file} | perl {czeng17_script_path} | cut -f4 > {en_file}'
        call(cs_cmd, debug)
        call(en_cmd, debug)
    else:
        print(f'already extracted: {en_file} and {cs_file}')
    return file

def extract_czeng17(extract_folder, debug=False):
    url = 'http://ufal.mff.cuni.cz/czeng/download.php?f=convert_czeng16_to_17.pl.zip'
    filename = f'{download_to}/convert_czeng16_to_17.pl.zip'
    extract_to = f'{extract_folder}/{get_extract_name(filename)}'
    script_path = f'{extract_to}/convert_czeng16_to_17.pl'
    
    if not os.path.exists(script_path):
        wget.download(url, filename, bar=bar_custom)  
        extract_to = extract_file(f'{download_to}/convert_czeng16_to_17.pl.zip', extract_folder, get_extract_name=get_extract_name, debug=debug)    
    return script_path

#########
# definitions of wmt data sources
# for es-en
# Punctuation in the official test sets will be encoded with ASCII characters (not complex Unicode characters) as much as possible. You may want to normalize your system's output before submission. You are able able to use a rawer version of the test sets that does not have this normalization.
# script to normalize punctuation: http://www.statmt.org/wmt11/normalize-punctuation.perl
wmt13_es_en = DLDataset(
    name='wmt13_es-en',
    train_urls=[
        'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
        'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'http://www.statmt.org/wmt13/training-parallel-un.tgz',
        'http://www.statmt.org/wmt13/training-parallel-nc-v8.tgz',
    ],
    valid_urls=[
        ('http://www.statmt.org/wmt13/dev.tgz', 'wmt13_dev.tgz')
    ],
    test_urls=[
        ('http://www.statmt.org/wmt13/test.tgz', 'wmt13_test.tgz')
    ],
    train_files_patterns=[
        ('*/europarl-v7.{src}-{tgt}.{lang}', ['es-en']), 
        ('*commoncrawl.{src}-{tgt}.{lang}', ['es-en']),
        ('*/news-commentary-v8.{src}-{tgt}.{lang}', ['es-en']),
        ('un/*undoc.2000.{src}-{tgt}.{lang}', ['es-en']), 
    ] ,
    valid_files_patterns=[
    ('dev/newstest2012.{lang}', ['es-en'])
    ],
    test_files_patterns=[
    ('test/newstest*.{lang}', ['es-en'])
    ],
)

wmt14_de_fr_en = DLDataset(
    name='wmt14_de_fr_en',
    train_urls=[
        'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
        'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'http://www.statmt.org/wmt13/training-parallel-un.tgz',
        'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
        ('http://www.statmt.org/wmt10/training-giga-fren.tar', 'training-giga-fren.gz.tar'), #it is actuall a gz.tar 
    ],
    valid_urls=[
        ('http://www.statmt.org/wmt14/dev.tgz', 'wmt14_dev.tgz'),
    ],
    test_urls=[
        ('http://www.statmt.org/wmt14/test-full.tgz', 'wmt14_test_full.tgz'), # cleaned test sets
    ],
    train_files_patterns=[
        ('*/europarl-v7.{src}-{tgt}.{lang}', ['fr-en', 'de-en']), 
        ('*commoncrawl.{src}-{tgt}.{lang}', ['fr-en', 'de-en']),
        ('*/*news-commentary-v9.{src}-{tgt}.{lang}', ['fr-en', 'de-en']),
        ('un/undoc.2000.{src}-{tgt}.{lang}', ['fr-en']),    
        ('*giga-{src}{tgt}*{lang}', ['fr-en'])
    ],
    valid_files_patterns=[
    ('dev/newstest2013.{lang}', ['fr-en', 'de-en'])
    ],
    test_files_patterns=[ 
    ('test-full/newstest*{src}{tgt}-{src:src}{tgt:ref}.{lang}', ['en-de', 'de-en', 'fr-en', 'en-fr']),                      
    ],
)

# pip install git+https://github.com/amake/tmx2corpus.git
wmt16_ro_en = DLDataset(
    name='wmt16_ro-en',
    train_urls=[
        ('http://data.statmt.org/wmt16/translation-task/training-parallel-ep-v8.tgz', 'wmt16_training-parallel-ep-v8.tgz'),
        ('http://opus.nlpl.eu/download.php?f=SETIMES/v2/tmx/en-ro.tmx.gz', 'en-ro.tmx.gz'),
    ],
    valid_urls=[
        ('http://data.statmt.org/wmt16/translation-task/dev-romanian-updated.tgz', 'wmt16_dev.tgz')
    ],
    test_urls=[
        ('http://data.statmt.org/wmt16/translation-task/test.tgz', 'wmt16_test.tgz')
    ],
    train_files_patterns=[
        ('*/*europarl-v8.{src}-{tgt}.{lang}', ['ro-en']), 
        ('bitext.{lang}', ['ro-en']) #setimes from tmux
        ] ,
    valid_files_patterns=[
    ('dev/newsdev2016*{src}{tgt}*.{lang}', ['ro-en', 'ro-en'])
    ],
    test_files_patterns=[
    ('test/newstest*{src}{tgt}*.{lang}', ['ro-en', 'en-ro'])
    ],
)

cwmt_wmt_instruction = 'cwmt download instruction at: http://nlp.nju.edu.cn/cwmt-wmt'
wmt17_fi_lv_tr_zh_en_manual_downloads = [
    # fake urls to have unique keys for the data
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/CASIA2015.zip', 'CASIA2015.zip'), cwmt_wmt_instruction),
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/CASICT2011.zip', 'CASICT2011.zip'), cwmt_wmt_instruction),
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/CASICT2015.zip', 'CASICT2015.zip'), cwmt_wmt_instruction),
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/Datum2015.zip', 'Datum2015.zip'), cwmt_wmt_instruction),
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/Datum2017.zip', 'Datum2017.zip'), cwmt_wmt_instruction),
    ( ('http://nlp.nju.edu.cn/cwmt-wmt/NEU2017.zip', 'NEU2017.zip'), cwmt_wmt_instruction),    
]
wmt17_fi_lv_tr_zh_en = DLDataset(
    name='wmt17_fi_lv_tr_zh_en',
    train_urls=[
        ('http://data.statmt.org/wmt17/translation-task/training-parallel-ep-v8.tgz', 'wmt17_training-parallel-ep-v8.tgz'),
        'http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
        'http://www.statmt.org/wmt15/wiki-titles.tgz',
        ('http://opus.nlpl.eu/download.php?f=SETIMES/v2/tmx/en-tr.tmx.gz', 'en-tr.tmx.gz'),
        ('http://data.statmt.org/wmt17/translation-task/rapid2016.tgz', 'wmt17_rapid2016.tgz'),
        'http://data.statmt.org/wmt17/translation-task/leta.v1.tgz',
        'http://data.statmt.org/wmt17/translation-task/dcep.lv-en.v1.tgz',
        'http://data.statmt.org/wmt17/translation-task/books.lv-en.v1.tgz',
        (('https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.00',
        'https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.01',), 'UNv1.0.en-zh.tar.gz'),
        #manually download files:
        ('http://nlp.nju.edu.cn/cwmt-wmt/CASIA2015.zip', 'CASIA2015.zip'),  
        ('http://nlp.nju.edu.cn/cwmt-wmt/CASICT2011.zip', 'CASICT2011.zip'),  
        ('http://nlp.nju.edu.cn/cwmt-wmt/CASICT2015.zip', 'CASICT2015.zip'),  
        ('http://nlp.nju.edu.cn/cwmt-wmt/Datum2015.zip', 'Datum2015.zip'), 
        ('http://nlp.nju.edu.cn/cwmt-wmt/Datum2017.zip', 'Datum2017.zip'),  
        ('http://nlp.nju.edu.cn/cwmt-wmt/NEU2017.zip', 'NEU2017.zip'),          
    ],
    valid_urls=[
        ('http://data.statmt.org/wmt17/translation-task/dev.tgz', 'wmt17_dev.tgz'),
    ],
    test_urls=[
        #NEW: Improved translations for zh test sets
        ('http://data.statmt.org/wmt17/translation-task/test-update-1.tgz', 'wmt17_test_zh_en.tgz'),    
        ('http://data.statmt.org/wmt17/translation-task/test.tgz', 'wmt17_test_others.tgz')
    ],
    train_files_patterns=[
        ('casict*/cas*{src:ch}{tgt:en}.txt', ['zh-en', 'zh-en'] ),
        ('casia*/cas*{src:ch}{tgt:en}.txt', ['zh-en', 'zh-en'] ),
        ('dataum*/Book*{src:cn}{tgt:en}.txt', ['zh-en', 'zh-en']),
        ('neu*/NEU*{src:cn}{tgt:en}.txt', ['zh-en', 'zh-en'] ),
        ('*/*UNv1.0.en-zh.{src:zh}{tgt:en}', ['zh-en']),
        ('training/*news-commentary-v12.{src}-{tgt}.{lang}', ['zh-en', ]),
        
        ('*/*europarl-v8.{src}-{tgt}.{lang}', ['fi-en', 'lv-en']),
        ('wiki/fi-en/titles.{src}-{tgt}.{lang}', ['fi-en', ]),  
        ('rapid2016.{tgt}-{src}.{lang}', ['fi-en', 'lv-en']),
        ('*/leta.{lang}', ['lv-en']),
        ('*/dcep.{lang}', ['lv-en']),
        ('*/farewell.{lang}', ['lv-en']),       
        ('bitext.{lang}', ['tr-en']),
    ] ,
    valid_files_patterns=[
    ('dev/newsdev2017*{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    [
        'fi-en', 'lv-en', 'tr-en', 'zh-en',
        'en-fi', 'en-lv', 'en-tr', 'en-zh'
    ]),                      
    ('dev/newstest2016*{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    [
        'fi-en',  'tr-en',  
        'en-fi',  'en-tr',  
    ]),  
    ],
    test_files_patterns=[
    ('test/newstest2017-{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    [
        'fi-en', 'lv-en', 'tr-en', 
        'en-fi', 'en-lv', 'en-tr',  
    ]),
    ('newstest2017-{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    [
        'zh-en',
        'en-zh'
    ]),
    ],
)

czeng_instruction = 'download instruction at: http://ufal.mff.cuni.cz/czeng/czeng16'
#alternative: use the prepared data but detokenize it?
wmt18_cs_et_en_manual_downloads = [
#for cs, need to register and download; Register and download CzEng 1.6.  
#Better results can be obtained by using a subset of sentences, released under a new version name CzEng 1.7.
    # ((f'http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.{i}.tar', 
    #     f'data-plaintext-format.{i}.tar'), czeng_instruction)
    # for i in range(10)
]

wmt18_cs_et_en = DLDataset(
    name='wmt18_cs_et_en',
    train_urls=[
        'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
        'http://data.statmt.org/wmt18/translation-task/training-parallel-ep-v8.tgz',
        'https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-cs.zipporah0-dedup-clean.tgz',
        'https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-et.zipporah0-dedup-clean.tgz',
        'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz',
        ('http://data.statmt.org/wmt18/translation-task/rapid2016.tgz', 'wmt18_rapid2016.tgz'),
        # (tuple(
        #     (f'http://ufallab.ms.mff.cuni.cz/~bojar/czeng16-data/data-plaintext-format.{i}.tar', 
        #     f'data-plaintext-format.{i}.tar')
        #     for i in range(10)
        # ), 
        # 'czeng16_data_plaintext.gz.tar'), 
    ],
    valid_urls=[
        ('http://data.statmt.org/wmt18/translation-task/dev.tgz', 'wmt18_dev.tgz'),
    ],
    test_urls=[
        ('http://data.statmt.org/wmt18/translation-task/test.tgz', 'wmt18_test.tgz'),
    ],
    train_files_patterns=[
        # ('*/*europarl-v7.{src}-{tgt}.{lang}', ['cs-en']),
        ('*/*europarl-v8.{src}-{tgt}.{lang}', ['et-en']),
        # ('*paracrawl-release1.{tgt}-{src}.zipporah0-dedup-clean.{lang}', ['cs-en', 'et-en']),
        ('*paracrawl-release1.{tgt}-{src}.zipporah0-dedup-clean.{lang}', ['et-en']),
        # ('*commoncrawl.{src}-{tgt}.{lang}', ['cs-en']),
        # ('*/news-commentary-v13.{src}-{tgt}.{lang}', ['cs-en']),
        # ('data.plaintext-format/*train.{lang}', ['cs-en']),
        ('rapid2016.{tgt}-{src}.{lang}', ['et-en']),
    ] ,
    valid_files_patterns=[
    ('dev/newsdev2018*{src}{tgt}-{src:src}{tgt:ref}.{lang}', ['et-en']),
    # ('dev/newstest2017*{src}{tgt}-{src:src}{tgt:ref}.{lang}', ['cs-en'])        
    ],
    test_files_patterns=[
    ('test/newstest2018-{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    # ['cs-en', 'et-en']),
    ['et-en']),
    ]
)

ru_en_yandex_instruction = 'Yandex Corpus download instruction at: https://translate.yandex.ru/corpus?lang=en'
wmt19_ru_gu_kk_lt_manual_downloads = [
    (('https://translate.yandex.ru/corpus?lang=en', 'wmt19_1mcorpus.zip'), ru_en_yandex_instruction)
]
wmt19_ru_gu_kk_lt = DLDataset(
    name='wmt19_ru_gu_kk_lt',
    train_urls=[
        'http://www.statmt.org/europarl/v9/training/europarl-v9.lt-en.tsv.gz',
        'https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-lt.bicleaner07.tmx.gz',
        'https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz',
        'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14-wmt19.en-kk.tsv.gz',
        'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-ru.tsv.gz',
        'http://data.statmt.org/wikititles/v1/wikititles-v1.kk-en.tsv.gz',
        'http://data.statmt.org/wikititles/v1/wikititles-v1.ru-en.tsv.gz',
        'http://data.statmt.org/wikititles/v1/wikititles-v1.kk-en.tsv.gz',
        'http://data.statmt.org/wikititles/v1/wikititles-v1.lt-en.tsv.gz',
        'http://data.statmt.org/wikititles/v1/wikititles-v1.gu-en.tsv.gz',
        (('https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.00',
        'https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.01',
        'https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.02',), 
        'wmt19_UNv1.0.en-ru.tar.gz'),
        'https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2016.en-lt.tmx.zip',
        ('https://translate.yandex.ru/corpus?lang=en', 'wmt19_1mcorpus.zip'),
    ],
    valid_urls=[
        ('http://data.statmt.org/wmt19/translation-task/dev.tgz', 'wmt19_dev.tgz'),
    ],
    test_urls=[
        ('http://data.statmt.org/wmt19/translation-task/test.tgz', 'wmt19_test.tgz'),
    ],
    train_files_patterns=[
        ('*europarl-v9.{src}-{tgt}.tsv.{lang}', ['lt-en']),
        #paracrawl
        ('*paracrawl-release1.{tgt}-{src}.zipporah0-dedup-clean.{lang}', ['ru-en']),
        ('bitext.{lang}', ['lt-en',]),
        ('*commoncrawl.{src}-{tgt}.{lang}', ['ru-en',]),
        ('*news-commentary-v14-wmt19.{tgt}-{src}.tsv.{lang}', ['kk-en', ]),
        ('*news-commentary-v14.{tgt}-{src}.tsv.{lang}', ['ru-en']),
        #yandex
        ('corpus.{tgt}_{src}.1m.{lang}', ['ru-en']),
        ('wikititles_v1_wikititles-v1.{src}-{tgt}.tsv.{lang}', ['ru-en', 'kk-en', 'lt-en', 'gu-en']),
        ('*/UNv1.0.{tgt}-{src}.{lang}', ['ru-en']),
        #rapid
        ('bitext.{lang}', ['lt-en'])
    ],
    valid_files_patterns=[
    ('dev/newsdev2019*{src}{tgt}-{src:src}{tgt:ref}.{lang}', ['gu-en', 'kk-en', 'lt-en']),
    ('dev/newstest2018*{src}{tgt}-{src:src}{tgt:ref}.{lang}', ['ru-en']),       
    ],
    test_files_patterns=[
    ('sgm/newstest2019-{src}{tgt}-{src:src}{tgt:ref}.{lang}', 
    ['ru-en', 'gu-en', 'kk-en', 'lt-en', 'en-ru', 'en-gu', 'en-kk', 'en-lt']),
    ]    
)


#########

if __name__ == "__main__":
    # speed up the downloads with multiple processing
    dl_folder = f'{to_data_path}/downloads'
    extract_folder = f'{to_data_path}/extracted'

    urls = [
        url
        for dataset in [wmt13_es_en, wmt14_de_fr_en, wmt16_ro_en, wmt18_cs_et_en, wmt19_ru_gu_kk_lt]
        for urls in [dataset.train_urls, dataset.valid_urls, dataset.test_urls]
        for url in urls
    ]
    urls = set(urls)
    download_multi(dl_folder, extract_folder, urls, num_processes=8, debug=True)

    # check manually downlaods
    to_manually_download_urls = (
        wmt17_fi_lv_tr_zh_en_manual_downloads + wmt18_cs_et_en_manual_downloads + wmt19_ru_gu_kk_lt_manual_downloads
    )
    to_be_manually_dowloaded = check_need_manual_downalod(dl_folder, to_manually_download_urls)
    if len(to_be_manually_dowloaded) > 0:
        print('Missing files that need to be downloaded manually; stop the process now.')
        exit(-1)
    
    completed_urls = {}
    completed_extraction = {}
    def work_on_wmt(directions, wmt_data):
        download_and_extract(
            to_data_path, 
            directions, 
            wmt_data, 
            to_manually_download_urls=to_manually_download_urls,
            completed_urls=completed_urls, completed_extraction=completed_extraction, debug=True)
                
    work_on_wmt(
        ['es_XX-en_XX'], 
        wmt13_es_en,)
    work_on_wmt(
        [
            'fr_XX-en_XX',  'en_XX-fr_XX',
            # 'en_XX-de_DE', 'de_DE-en_XX',
        ], 
        wmt14_de_fr_en,)
    work_on_wmt(
        ['ro_RO-en_XX', 'en_XX-ro_XX'], 
        wmt16_ro_en,)
    work_on_wmt(
        [
            # 'zh_CN-en_XX', 
            'lv_LV-en_XX', 'fi_FI-en_XX', 'tr_TR-en_XX',
            #in case the reversed directions have different train/valid/test data
            # 'en_XX-zh_CN', 
            'en_XX-lv_LV', 'en_XX-fi_FI', 'en_XX-tr_TR',
        ], 
        wmt17_fi_lv_tr_zh_en, )
    # czeng17_script_path = download_czeng17_script(download_to, extract_to, debug=False)
    # cz_username =  None
    work_on_wmt(
        [
            # 'cs_CZ-en_XX', 
            'et_EE-en_XX'], 
        wmt18_cs_et_en,)
    work_on_wmt(
        [
            # 'ru_RU-en_XX', 'en_XX-ru_RU', 
            'gu_IN-en_XX', 'kk_KZ-en_XX', 'lt_LT-en_XX',
            #in case the reversed directions have different train/valid/test data
            'en_XX-gu_IN', 'en_XX-kk_KZ', 'en_XX-lt_LT'
        ], 
        wmt19_ru_gu_kk_lt,)

    not_matching = check_wmt_test_bleu(
        f'{to_data_path}/raw', 
        [
            ('wmt13', ['es_XX-en_XX']),
            ('wmt14/full', ['fr_XX-en_XX',]),
            ('wmt16', ['ro_RO-en_XX',]),
            # ('wmt17/improved', ['zh_CN-en_XX']),
            ('wmt17', [ 'lv_LV-en_XX', 'fi_FI-en_XX', 'tr_TR-en_XX']),
            ('wmt18', ['cs_CZ-en_XX', 'et_EE-en_XX']),
            ('wmt19', ['gu_IN-en_XX', 'kk_KZ-en_XX', 'lt_LT-en_XX']), 
            #'ru_RU-en_XX', 
        ]
        )    
    if len(not_matching) > 0:
        print('the following datasets do not have matching test datasets:\n\t', '\n\t'.join(not_matching))

