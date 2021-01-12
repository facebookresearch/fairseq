# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import os
import csv
from collections import defaultdict
from six.moves import zip
import io
import wget
import sys

from subprocess import check_call, check_output

# scripts and data locations
CWD = os.getcwd()
UTILS = f"{CWD}/utils"

MOSES = f"{UTILS}/mosesdecoder"

WORKDIR_ROOT = os.environ.get('WORKDIR_ROOT', None)

if WORKDIR_ROOT is None or  not WORKDIR_ROOT.strip():
    print('please specify your working directory root in OS environment variable WORKDIR_ROOT. Exitting..."')
    sys.exit(-1)


# please donwload mosesdecoder here:
detok_cmd = f'{MOSES}/scripts/tokenizer/detokenizer.perl'


def call(cmd):
    print(f"Executing: {cmd}")
    check_call(cmd, shell=True)

class MultiLingualAlignedCorpusReader(object):
    """A class to read TED talk dataset
    """

    def __init__(self, corpus_path, delimiter='\t',
                 target_token=True, bilingual=True, corpus_type='file',
                 lang_dict={'source': ['fr'], 'target': ['en']},
                 eval_lang_dict=None, zero_shot=False,
                 detok=True,
                 ):

        self.empty_line_flag = 'NULL'
        self.corpus_path = corpus_path
        self.delimiter = delimiter
        self.bilingual = bilingual
        self.lang_dict = lang_dict
        self.lang_set = set()
        self.target_token = target_token
        self.zero_shot = zero_shot
        self.eval_lang_dict = eval_lang_dict
        self.corpus_type = corpus_type
        self.detok = detok

        for list_ in self.lang_dict.values():
            for lang in list_:
                self.lang_set.add(lang)

        self.data = dict()
        self.data['train'] = self.read_aligned_corpus(split_type='train')
        self.data['test'] = self.read_aligned_corpus(split_type='test')
        self.data['dev'] = self.read_aligned_corpus(split_type='dev')

    def read_data(self, file_loc_):
        data_list = list()
        with io.open(file_loc_, 'r', encoding='utf8') as fp:
            for line in fp:
                try:
                    text = line.strip()
                except IndexError:
                    text = self.empty_line_flag
                data_list.append(text)
        return data_list

    def filter_text(self, dict_):
        if self.target_token:
            field_index = 1
        else:
            field_index = 0
        data_dict = defaultdict(list)
        list1 = dict_['source']
        list2 = dict_['target']
        for sent1, sent2 in zip(list1, list2):
            try:
                src_sent = ' '.join(sent1.split()[field_index: ])
            except IndexError:
                src_sent = 'NULL'

            if src_sent.find(self.empty_line_flag) != -1 or len(src_sent) == 0:
                continue

            elif sent2.find(self.empty_line_flag) != -1 or len(sent2) == 0:
                continue

            else:
                data_dict['source'].append(sent1)
                data_dict['target'].append(sent2)
        return data_dict

    def read_file(self, split_type, data_type):
        return self.data[split_type][data_type]

    def save_file(self, path_, split_type, data_type, lang):
        tok_file = tok_file_name(path_, lang)
        with io.open(tok_file, 'w', encoding='utf8') as fp:
            for line in self.data[split_type][data_type]:
                fp.write(line + '\n')
        if self.detok:
            de_tok(tok_file, lang)                

    def add_target_token(self, list_, lang_id):
        new_list = list()
        token = '__' + lang_id + '__'
        for sent in list_:
            new_list.append(token + ' ' + sent)
        return new_list

    def read_from_single_file(self, path_, s_lang, t_lang):
        data_dict = defaultdict(list)
        with io.open(path_, 'r', encoding='utf8') as fp:
            reader = csv.DictReader(fp, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                data_dict['source'].append(row[s_lang])
                data_dict['target'].append(row[t_lang])

        if self.target_token:
            text = self.add_target_token(data_dict['source'], t_lang)
            data_dict['source'] = text

        return data_dict['source'], data_dict['target']

    def read_aligned_corpus(self, split_type='train'):
        data_dict = defaultdict(list)
        iterable = []
        s_list = []
        t_list = []

        if self.zero_shot:
            if split_type == "train":
                iterable = zip(self.lang_dict['source'], self.lang_dict['target'])
            else:
                iterable = zip(self.eval_lang_dict['source'], self.eval_lang_dict['target'])

        elif self.bilingual:
            iterable = itertools.product(self.lang_dict['source'], self.lang_dict['target'])

        for s_lang, t_lang in iterable:
            if s_lang == t_lang:
                continue
            if self.corpus_type == 'file':
                split_type_file_path = os.path.join(self.corpus_path,
                                                    "all_talks_{}.tsv".format(split_type))
                s_list, t_list = self.read_from_single_file(split_type_file_path,
                                                            s_lang=s_lang,
                                                            t_lang=t_lang)
            data_dict['source'] += s_list
            data_dict['target'] += t_list
        new_data_dict = self.filter_text(data_dict)
        return new_data_dict


def read_langs(corpus_path):
    split_type_file_path = os.path.join(corpus_path, 'extracted',
                                        "all_talks_dev.tsv")    
    with io.open(split_type_file_path, 'r', encoding='utf8') as fp:
        reader = csv.DictReader(fp, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader)
        return [k for k in header.keys() if k != 'talk_name']

def extra_english(corpus_path, split):
    split_type_file_path = os.path.join(corpus_path,
                                        f"all_talks_{split}.tsv") 
    output_split_type_file_path = os.path.join(corpus_path,
                                        f"all_talks_{split}.en")                                            
    with io.open(split_type_file_path, 'r', encoding='utf8') as fp, io.open(output_split_type_file_path, 'w', encoding='utf8') as fw:
        reader = csv.DictReader(fp, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            line = row['en']
            fw.write(line + '\n')
    de_tok(output_split_type_file_path, 'en')



def tok_file_name(filename, lang):
    seps = filename.split('.')
    seps.insert(-1, 'tok')
    tok_file = '.'.join(seps)
    return tok_file

def de_tok(tok_file, lang):
    # seps = tok_file.split('.')
    # seps.insert(-1, 'detok')
    # de_tok_file = '.'.join(seps)
    de_tok_file = tok_file.replace('.tok.', '.')
    cmd = 'perl {detok_cmd} -l {lang} < {tok_file} > {de_tok_file}'.format(
        detok_cmd=detok_cmd, tok_file=tok_file,
        de_tok_file=de_tok_file, lang=lang[:2])
    call(cmd)

def extra_bitex(
    ted_data_path,
    lsrc_lang,
    ltrg_lang,
    target_token,
    output_data_path,
):
    def get_ted_lang(lang):
        long_langs = ['pt-br', 'zh-cn', 'zh-tw', 'fr-ca']
        if lang[:5] in long_langs:
            return lang[:5]
        elif lang[:4] =='calv':
            return lang[:5]
        elif lang in ['pt_BR', 'zh_CN', 'zh_TW', 'fr_CA']:
            return lang.lower().replace('_', '-')
        return lang[:2]
    src_lang = get_ted_lang(lsrc_lang)
    trg_lang = get_ted_lang(ltrg_lang)
    train_lang_dict={'source': [src_lang], 'target': [trg_lang]}
    eval_lang_dict = {'source': [src_lang], 'target': [trg_lang]}

    obj = MultiLingualAlignedCorpusReader(corpus_path=ted_data_path,
                                          lang_dict=train_lang_dict,
                                          target_token=target_token,
                                          corpus_type='file',
                                          eval_lang_dict=eval_lang_dict,
                                          zero_shot=False,
                                          bilingual=True)

    os.makedirs(output_data_path, exist_ok=True)
    lsrc_lang = lsrc_lang.replace('-', '_')
    ltrg_lang = ltrg_lang.replace('-', '_')
    obj.save_file(output_data_path + f"/train.{lsrc_lang}-{ltrg_lang}.{lsrc_lang}",
                  split_type='train', data_type='source', lang=src_lang)
    obj.save_file(output_data_path + f"/train.{lsrc_lang}-{ltrg_lang}.{ltrg_lang}",
                  split_type='train', data_type='target', lang=trg_lang)

    obj.save_file(output_data_path + f"/test.{lsrc_lang}-{ltrg_lang}.{lsrc_lang}",
                  split_type='test', data_type='source', lang=src_lang)
    obj.save_file(output_data_path + f"/test.{lsrc_lang}-{ltrg_lang}.{ltrg_lang}",
                  split_type='test', data_type='target', lang=trg_lang)

    obj.save_file(output_data_path + f"/valid.{lsrc_lang}-{ltrg_lang}.{lsrc_lang}",
                  split_type='dev', data_type='source', lang=src_lang)
    obj.save_file(output_data_path + f"/valid.{lsrc_lang}-{ltrg_lang}.{ltrg_lang}",
                  split_type='dev', data_type='target', lang=trg_lang)


def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] Ks" % (current / total * 100, current / 1000, total / 1000), end='\r')


def download_and_extract(download_to, extract_to):
    url = 'http://phontron.com/data/ted_talks.tar.gz'
    filename = f"{download_to}/ted_talks.tar.gz"
    if os.path.exists(filename):
        print(f'{filename} has already been downloaded so skip')
    else:
        filename = wget.download(url, filename, bar=bar_custom)
    if os.path.exists(f'{extract_to}/all_talks_train.tsv'):
        print(f'Already extracted so skip')
    else:
        extract_cmd = f'tar xzfv "{filename}" -C "{extract_to}"'
        call(extract_cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ted_data_path', type=str, default=WORKDIR_ROOT, required=False)
    parser.add_argument(
        '--direction-list', 
        type=str, 
        # default=None,
        #for ML50
        default=(
            "bn_IN-en_XX,he_IL-en_XX,fa_IR-en_XX,id_ID-en_XX,sv_SE-en_XX,pt_XX-en_XX,ka_GE-en_XX,ka_GE-en_XX,th_TH-en_XX,"
            "mr_IN-en_XX,hr_HR-en_XX,uk_UA-en_XX,az_AZ-en_XX,mk_MK-en_XX,gl_ES-en_XX,sl_SI-en_XX,mn_MN-en_XX,"
            #non-english directions
            # "fr_XX-de_DE," # replaced with wmt20
            # "ja_XX-ko_KR,es_XX-pt_XX,ru_RU-sv_SE,hi_IN-bn_IN,id_ID-ar_AR,cs_CZ-pl_PL,ar_AR-tr_TR"
        ), 
        required=False)
    parser.add_argument('--target-token',  action='store_true', default=False)
    parser.add_argument('--extract-all-english',  action='store_true', default=False)    

    args = parser.parse_args()

    import sys
    import json

    # TED Talks data directory
    ted_data_path = args.ted_data_path

    download_to = f'{ted_data_path}/downloads'
    extract_to = f'{ted_data_path}/extracted'
    
    #DESTDIR=${WORKDIR_ROOT}/ML50/raw/
    output_path = f'{ted_data_path}/ML50/raw'
    os.makedirs(download_to, exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    download_and_extract(download_to, extract_to)        


    if args.extract_all_english:
        for split in ['train', 'dev', 'test']:
            extra_english(ted_data_path, split)
        exit(0)     
    if args.direction_list is not None: 
        directions = args.direction_list.strip().split(',')
        directions = [tuple(d.strip().split('-', 1)) for d in directions if d]
    else: 
        langs = read_langs(ted_data_path)
        # directions = [
        #     '{}.{}'.format(src, tgt) 
        #     for src in langs 
        #     for tgt in langs
        #     if src < tgt
        # ]
        directions = [('en', tgt) for tgt in langs if tgt != 'en']
    print(f'num directions={len(directions)}: {directions}')

    for src_lang, trg_lang in directions:
        print('--working on {}-{}'.format(src_lang, trg_lang))
        extra_bitex(
            extract_to,
            src_lang,
            trg_lang,
            target_token=args.target_token,
            output_data_path=output_path
        )
