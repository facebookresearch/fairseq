# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import os
import itertools
import shutil
import tarfile
import zipfile
import requests
from subprocess import check_call

"""
Dependencies:
MOSES
git clone https://github.com/moses-smt/mosesdecoder.git
"""

if os.getenv('MOSES') is not None:
    SGM_TOOL = f"{os.getenv('MOSES')}/scripts/ems/support/input-from-sgm.perl"
else:
    raise TypeError('Setting environement variable MOSESE is required for formatting WMT test files')


def xml2txt(xml, output_file_path):
    cmd = "\\\n".join([
        f"grep '<seg id' {xml} | ",
        "sed -e 's/<seg id=\"[0-9]*\">\\s*//g' | ",
        "sed -e 's/\\s*<\\/seg>\\s*//g' | ",
        f"sed -e \"s/\\’/\\'/g\" > {output_file_path}"
    ])
    check_call(cmd, shell=True)
    return True


def sgm2raw(sgm, output_file_path):
    to_file = sgm[0:len(sgm) - len('.sgm')]
    cmd = f'{SGM_TOOL} < {sgm} > {to_file}'
    check_call(cmd, shell=True)
    shutil.copy(to_file, output_file_path)
    return True


def get_tsv_column(tsvfile, column_index, output_file):
    cmd = (
        "awk -F'\t' '{print $" + str(column_index) + "}' " +
        f"{tsvfile} | tail -n +2 > {output_file}"
    )
    check_call(cmd, shell=True)


def download_file(download_url, download_path):
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url}!")
        return False
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")
    return True


def untar(tarfile_path, extract_to):
    print(f'Untarring {tarfile_path} into {extract_to}')
    with tarfile.open(tarfile_path) as tar:
        tar.extractall(extract_to)


def unzip(zipfile_path, extract_to):
    print(f'Unzipping {zipfile_path} into {extract_to}')
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def download_wat(directory):
    dataset_directory = os.path.join(directory, "wat")
    os.makedirs(dataset_directory, exist_ok=True)
    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    print("Saving WAT test data to:", dataset_directory)

    myen_download_url = "http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/wat2020.my-en.zip"
    myen_download_path = f"{tmp_directory}/myen.zip"

    kmen_download_url = "http://lotus.kuee.kyoto-u.ac.jp/WAT/km-en-data/wat2020.km-en.zip"
    kmen_download_path = f"{tmp_directory}/kmen.zip"

    hien_download_url = "http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/dev_test.tgz"
    hien_download_path = f"{tmp_directory}/hien.tgz"

    download_file(myen_download_url, myen_download_path)
    download_file(kmen_download_url, kmen_download_path)
    download_file(hien_download_url, hien_download_path)

    unzip(myen_download_path, tmp_directory)
    unzip(kmen_download_path, tmp_directory)
    untar(hien_download_path, tmp_directory)

    shutil.move(
        os.path.join(tmp_directory, 'wat2020.km-en/alt', 'test.alt.km'),
        os.path.join(dataset_directory, 'test.eng_Latn-khm_Khmr.khm_Khmr')
    )
    shutil.move(
        os.path.join(tmp_directory, 'wat2020.km-en/alt', 'test.alt.en'),
        os.path.join(dataset_directory, 'test.eng_Latn-khm_Khmr.eng_Latn')
    )
    shutil.move(
        os.path.join(tmp_directory, 'wat2020.my-en/alt', 'test.alt.my'),
        os.path.join(dataset_directory, 'test.eng_Latn-mya_Mymr.mya_Mymr')
    )
    shutil.move(
        os.path.join(tmp_directory, 'wat2020.my-en/alt', 'test.alt.en'),
        os.path.join(dataset_directory, 'test.eng_Latn-mya_Mymr.eng_Latn')
    )
    shutil.move(
        os.path.join(tmp_directory, 'dev_test', 'test.en'),
        os.path.join(dataset_directory, 'test.eng_Latn-hin_Deva.eng_Latn')
    )
    shutil.move(
        os.path.join(tmp_directory, 'dev_test', 'test.hi'),
        os.path.join(dataset_directory, 'test.eng_Latn-hin_Deva.hin_Deva')
    )

    shutil.rmtree(tmp_directory)


def download_floresv1(directory):
    # Dowloand data
    dataset_directory = os.path.join(directory, "floresv1")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Flores(v1) test data to:", dataset_directory)

    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    download_url = "https://github.com/facebookresearch/flores/raw/main/previous_releases/floresv1/data/flores_test_sets.tgz"
    download_path = f"{tmp_directory}/floresv1.tgz"

    download_file(download_url, download_path)
    untar(download_path, tmp_directory)

    langs = ['km', 'ps', 'si']
    code_langs = ['khm_Khmr', 'pbt_Arab', 'sin_Sinh']
    for lang, code_lang in zip(langs, code_langs):
        shutil.move(
            os.path.join(tmp_directory, 'flores_test_sets', f'wikipedia.devtest.{lang}-en.{lang}'),
            os.path.join(dataset_directory, f'test.eng_Latn-{code_lang}.{code_lang}')
        )
        shutil.move(
            os.path.join(tmp_directory, 'flores_test_sets', f'wikipedia.devtest.{lang}-en.en'),
            os.path.join(dataset_directory, f'test.eng_Latn-{code_lang}.eng_Latn')
        )
    shutil.rmtree(tmp_directory)


def download_wmt(directory):
    dataset_directory = os.path.join(directory, "wmt")
    os.makedirs(dataset_directory, exist_ok=True)
    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    urls = [
        "http://www.statmt.org/wmt13/test.tgz",
        "http://www.statmt.org/wmt14/test-full.tgz",
        "http://data.statmt.org/wmt16/translation-task/test.tgz",
        "http://data.statmt.org/wmt17/translation-task/test.tgz",
        "http://data.statmt.org/wmt18/translation-task/test.tgz",
        "http://data.statmt.org/wmt19/translation-task/test.tgz",
    ]
    paths = [
        'wmt13',
        'wmt14',  # incorrect order
        'wmt16',
        'wmt17',
        'wmt18',
        'wmt19'
    ]

    list_sgmfiles = [
        ['test/newstest2013-src.es.sgm', 'test/newstest2013-src.en.sgm'],
        list(itertools.chain.from_iterable([[
            f'test-full/newstest2014-{lang}en-src.{lang}.sgm',
            f'test-full/newstest2014-{lang}en-ref.en.sgm'
        ] for lang in ['de', 'fr', 'hi']])),
        ['test/newstest2016-enro-src.en.sgm', 'test/newstest2016-enro-ref.ro.sgm'],
        ['test/newstest2017-lven-src.lv.sgm', 'test/newstest2017-lven-ref.en.sgm'],
        list(itertools.chain.from_iterable([[
            f'test/newstest2018-{lang}en-src.{lang}.sgm',
            f'test/newstest2018-{lang}en-ref.en.sgm'
        ] for lang in ['cs', 'et', 'tr']])),
        list(itertools.chain.from_iterable([[
            f'sgm/newstest2019-{lang}en-src.{lang}.sgm',
            f'sgm/newstest2019-{lang}en-ref.en.sgm'
        ] for lang in ['fi', 'gu', 'kk', 'lt', 'ru', 'zh']]))
    ]
    list_rawfiles = [
        ['test.eng_Latn-spa_Latn.spa_Latn', 'test.eng_Latn-spa_Latn.eng_Latn'],
        ['test.deu_Latn-eng_Latn.deu_Latn', 'test.deu_Latn-eng_Latn.eng_Latn',
         'test.eng_Latn-fra_Latn.fra_Latn',
         'test.eng_Latn-fra_Latn.eng_Latn', 'test.eng_Latn-hin_Deva.hin_Deva',
         'test.eng_Latn-hin_Deva.eng_Latn'],
        ["test.eng_Latn-ron_Latn.eng_Latn", "test.eng_Latn-ron_Latn.ron_Latn"],
        ['test.eng_Latn-lvs_Latn.lvs_Latn', 'test.eng_Latn-lvs_Latn.eng_Latn'],
        ['test.ces_Latn-eng_Latn.ces_Latn', 'test.ces_Latn-eng_Latn.eng_Latn',
         'test.eng_Latn-est_Latn.est_Latn',
         'test.eng_Latn-est_Latn.eng_Latn', 'test.eng_Latn-tur_Latn.tur_Latn',
         'test.eng_Latn-tur_Latn.eng_Latn'],
        list(itertools.chain.from_iterable([[
            f'test.eng_Latn-{lang}.{lang}',
            f'test.eng_Latn-{lang}.eng_Latn'
        ] for lang in ['fin_Latn', 'guj_Gujr', 'kaz_Cyrl', 'lit_Latn', 'rus_Cyrl', 'zho_Hans']]))
    ]

    for url, path, sgmfiles, rawfiles in zip(urls, paths, list_sgmfiles, list_rawfiles):
        tgz_file = os.path.join(tmp_directory, f'{path}.tgz')
        download_file(url, tgz_file)
        extracted_dir = os.path.join(tmp_directory, path)
        os.makedirs(extracted_dir, exist_ok=True)
        untar(tgz_file, extracted_dir)
        for sgmfile, rawfile in zip(sgmfiles, rawfiles):
            sgm2raw(
                os.path.join(extracted_dir, sgmfile),
                os.path.join(dataset_directory, rawfile),
            )
    shutil.rmtree(tmp_directory)


def download_autshumato(directory, pre_download_directory):
    # Dowloand data
    dataset_directory = os.path.join(directory, "autshumato")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Autshumato test data to:", dataset_directory)

    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    download_path = os.path.join(pre_download_directory, 'Autshumato_MT_Evaluation_Set.zip')
    unzip(download_path, tmp_directory)

    langs = [
        "Afrikaans",
        "IsiXhosa",
        "IsiZulu",
        "Sepedi",
        "Sesotho",
        "Setswana",
        "Siswati",
        "Xitsonga",
    ]

    codes = ['afr_Latn', 'xho_Latn', 'zul_Latn', 'nso_Latn', 'sot_Latn', 'tsn_Latn', 'ssw_Latn',  'tso_Latn']
    filename = os.path.join(tmp_directory, "Autshumato_MT_Evaluation_Set", "Autshumato.EvaluationSet.{}.Translator{}.txt")
    for lang, code in zip(langs, codes):
        test_src_out = open("{}/test.eng_Latn-{}.eng_Latn".format(dataset_directory, code), "w")
        test_tgt_out = open("{}/test.eng_Latn-{}.{}".format(dataset_directory, code, code), "w")

        dev_src_out = open("{}/dev.eng_Latn-{}.eng_Latn".format(dataset_directory, code), "w")
        dev_tgt_out = open("{}/dev.eng_Latn-{}.{}".format(dataset_directory, code, code), "w")

        for translator in range(4):
            src_in = filename.format('English', translator + 1)
            tgt_in = filename.format(lang, translator + 1)
            with open(src_in, 'r') as src, open(tgt_in, 'r') as tgt:
                # First half dev and second half test
                for e, (x, y) in enumerate(zip(src, tgt)):
                    if not x.startswith('<Doc'):
                        x = x.strip()
                        y = y.strip()
                        if e < 257:
                            dev_src_out.write(x + '\n')
                            dev_tgt_out.write(y + '\n')
                        else:
                            test_src_out.write(x + '\n')
                            test_tgt_out.write(y + '\n')
    shutil.rmtree(tmp_directory)


def download_tico(directory):
    dataset_directory = os.path.join(directory, "tico")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving TICO test data to:", dataset_directory)

    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    url = "https://tico-19.github.io/data/tico19-testset.zip"
    tgz_file = os.path.join(tmp_directory, 'tico.zip')
    download_file(url, tgz_file)
    unzip(tgz_file, tmp_directory)
    langs = ["am", "ar", "bn", "ckb", "es-LA", "fa", "fr", "fuv", "ha", "hi", "id",
             "ku", "ln", "lg", "mr", "ms", "my", "ne", "om", "prs", "ps", "pt-BR",
             "ru", "rw", "so", "sw", "ti", "tl", "ur", "zh", "zu"]
    code_langs = ["amh_Ethi", "arb_Arab", "ben_Beng", "ckb_Arab", "spa_Latn", "pes_Arab",
                  "fra_Latn", "fuv_Latn", "hau_Latn", "hin_Deva", "ind_Latn", "kmr_Latn",
                  "lin_Latn", "lug_Latn", "mar_Deva", "zsm_Latn", "mya_Mymr", "npi_Deva",
                  "gaz_Latn", "prs_Arab", "pbt_Arab", "por_Latn", "rus_Cyrl", "kin_Latn",
                  "som_Latn", "swh_Latn", "tir_Ethi", "tgl_Latn", "urd_Arab", "zho_Hans",
                  "zul_Latn"]
    for lang, code_lang in zip(langs, code_langs):
        tsvfile = f"{tmp_directory}/tico19-testset/test/test.en-{lang}.tsv"
        pair = f"eng_Latn-{code_lang}" if "eng_Latn" < code_lang else f"{code_lang}-eng_Latn"
        get_tsv_column(tsvfile, 3, f"{dataset_directory}/test.{pair}.eng_Latn")
        get_tsv_column(tsvfile, 4, f"{dataset_directory}/test.{pair}.{code_lang}")
    shutil.rmtree(tmp_directory)


def download_madar(directory, pre_download_directory):
    dataset_directory = os.path.join(directory, "madar")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving MADAR test data to:", dataset_directory)

    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)

    tgz_file = os.path.join(
        pre_download_directory,
        'MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip'
    )
    unzip(tgz_file, tmp_directory)
    madar_dir = os.path.join(
        tmp_directory,
        "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus"
    )
    cities = [
        'Alexandria', 'Amman', 'Aswan', 'Baghdad', 'Basra', 'Beirut',
        'Cairo', 'Fes', 'Jeddah', 'Mosul', 'Rabat', 'Riyadh', 'Salt',
        'Sanaa', 'Sfax', 'Tunis'
    ]
    flores = {
        'aeb_Arab': ['Tunis', 'Sfax'],
        'acm_Arab': ['Mosul', 'Baghdad', 'Basra'],
        'acq_Arab': ['Sanaa'],
        'ajp_Arab': ['Amman', 'Salt'],
        'apc_Arab': ['Beirut'],
        'ars_Arab': ['Riyadh', 'Jeddah'],
        'ary_Arab': ['Rabat', 'Fes'],
        'arz_Arab': ['Cairo', 'Alexandria', 'Aswan'],
    }

    # Using the corpus_6_test_corpus_26_test split as our test set
    MSA = {}
    MSA_path = f"{madar_dir}/MADAR.corpus.MSA.tsv"
    with open(MSA_path, 'r') as f:
        for line in f:
            id, split, lang, sent = line.strip().split('\t')
            if sent == 'sent':
                continue
            if split == 'corpus-6-test-corpus-26-test':
                MSA[id] = sent
    DIALECTS = {}
    for city in cities:
        city_path = f"{madar_dir}/MADAR.corpus.{city}.tsv"

        if city not in DIALECTS:
            DIALECTS[city] = {}

        with open(city_path, 'r') as f:
            for line in f:

                id, split, lang, sent = line.strip().split('\t')
                if sent == 'sent':
                    continue
                if split == 'corpus-6-test-corpus-26-test':
                    DIALECTS[city][id] = sent
            # Stats:

    for code in flores:
        for city in flores[code]:
            # City dict:
            src = open(os.path.join(dataset_directory, f"test.arb_Arab-{code}.arb_Arab"), 'a')
            tgt = open(os.path.join(dataset_directory, f"test.arb_Arab-{code}.{code}"), 'a')
            for id in DIALECTS[city]:
                tgt.write(DIALECTS[city][id] + '\n')
                src.write(MSA[id] + '\n')
            src.close()
            tgt.close()
    shutil.rmtree(tmp_directory)


def download_mafand(directory):
    dataset_directory = os.path.join(directory, "mafand")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving MAFAND-MT test data to:", dataset_directory)

    fra_directions = ['bam', 'ewe', 'fon', 'mos', 'wol']
    for tgt in fra_directions:
        tgt_nllb_code = f"{tgt}_Latn"
        pair = f"fra_Latn-{tgt_nllb_code}" if "fra_Latn" < tgt_nllb_code else f"{tgt_nllb_code}-fra_Latn"

        source_download_url = f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/fr_{tgt}_news/test.fr"
        source_download_path = f"{dataset_directory}/test.{pair}.fra_Latn"

        target_download_url = f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/fr_{tgt}_news/test.{tgt}"
        target_download_path = f"{dataset_directory}/test.{pair}.{tgt_nllb_code}"

        download_file(source_download_url, source_download_path)
        download_file(target_download_url, target_download_path)

    # Luo currently not available due to copyright issues.
    eng_directions = ['ibo', 'hau', 'lug', 'swa', 'tsn', 'twi', 'yor', 'zul']
    for tgt in eng_directions:
        tgt_nllb_code = "swh_Latn" if tgt == 'swa' else f"{tgt}_Latn"
        pair = f"eng_Latn-{tgt_nllb_code}" if "eng_Latn" < tgt_nllb_code else f"{tgt_nllb_code}-eng_Latn"

        source_download_url = f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/en_{tgt}_news/test.en"
        source_download_path = f"{dataset_directory}/test.{pair}.eng_Latn"

        target_download_url = f"https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/en_{tgt}_news/test.{tgt}"
        target_download_path = f"{dataset_directory}/test.{pair}.{tgt_nllb_code}"

        download_file(source_download_url, source_download_path)
        download_file(target_download_url, target_download_path)


def download_iwslt(directory, pre_download_directory):
    dataset_directory = os.path.join(directory, "iwslt")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving IWSLT test data to:", dataset_directory)

    tmp_directory = os.path.join(dataset_directory, "tmp")
    os.makedirs(tmp_directory, exist_ok=True)
    # source tgz files must be in args.pre_download:
    # Must download:
    # 2017-01-mted-test.tgz - https://wit3.fbk.eu/2017-01-b
    # 2017-01-ted-test.tgz  - https://wit3.fbk.eu/2017-01-d
    # 2015-01-test.tgz -  https://wit3.fbk.eu/2015-01-b
    # 2014-01-test.tgz -  https://wit3.fbk.eu/2014-01-b
    # 2014
    untar(os.path.join(pre_download_directory, '2014-01-test.tgz'), tmp_directory)
    map_langs = {'fa': "pes_Arab", 'pl': 'pol_Latn', 'ru': 'rus_Cyrl', 'vi': 'vie_Latn',
                 'ar': 'arb_Arab', 'de': 'deu_Latn', 'fr': 'fra_Latn', 'ja': 'jpn_Jpan',
                 'ko': 'kor_Hang', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'ro': 'ron_Latn'}
    for l in ['fa', 'pl', 'ru']:
        pair_dir = f"{tmp_directory}/2014-01-test/texts/en/{l}/"
        untar(pair_dir + f'en-{l}.tgz', pair_dir)
        src_xml = f"{pair_dir}/en-{l}/IWSLT14.TED.tst2014.en-{l}.en.xml"
        tgt_xml = f"{pair_dir}/en-{l}/IWSLT14.TED.tst2014.en-{l}.{l}.xml"
        src_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.eng_Latn"
        tgt_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.{map_langs[l]}"

        xml2txt(src_xml,  src_output)
        xml2txt(tgt_xml,  tgt_output)

    untar(os.path.join(pre_download_directory, '2015-01-test.tgz'), tmp_directory)
    for l in ['vi']:
        pair = f'en-{l}'
        pair_dir = f"{tmp_directory}/2015-01-test/texts/en/{l}/"
        untar(pair_dir + f'{pair}.tgz', pair_dir)
        src_xml = f"{pair_dir}/{pair}/IWSLT15.TED.tst2015.{pair}.en.xml"
        src_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.eng_Latn"
        xml2txt(src_xml,  src_output)

        pair = f"{l}-en"
        pair_dir = f"{tmp_directory}/2015-01-test/texts/{l}/en/"
        untar(pair_dir + f'{pair}.tgz', pair_dir)
        tgt_xml = f"{pair_dir}/{pair}/IWSLT15.TED.tst2015.{pair}.{l}.xml"
        tgt_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.{map_langs[l]}"
        xml2txt(tgt_xml,  tgt_output)

    untar(os.path.join(pre_download_directory, '2017-01-ted-test.tgz'), tmp_directory)
    for l in ['ar', 'de', 'fr', 'ja', 'ko']:
        pair_dir = f"{tmp_directory}/2017-01-ted-test/texts/en/{l}/"
        untar(pair_dir + f'en-{l}.tgz', pair_dir)
        rev_pair_dir = f"{tmp_directory}/2017-01-ted-test/texts/{l}/en/"
        untar(rev_pair_dir + f'{l}-en.tgz', rev_pair_dir)

        src_xml = f"{pair_dir}/en-{l}/IWSLT17.TED.tst2017.en-{l}.en.xml"
        tgt_xml = f"{rev_pair_dir}/{l}-en/IWSLT17.TED.tst2017.{l}-en.{l}.xml"
        src_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.eng_Latn"
        tgt_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.{map_langs[l]}"

        xml2txt(src_xml,  src_output)
        xml2txt(tgt_xml,  tgt_output)

    untar(os.path.join(pre_download_directory, '2017-01-mted-test.tgz'), tmp_directory)
    for l in ['it', 'nl', 'ro']:
        pair_dir = f"{tmp_directory}/2017-01-mted-test/texts/en/{l}/"
        untar(pair_dir + f'en-{l}.tgz', pair_dir)
        rev_pair_dir = f"{tmp_directory}/2017-01-mted-test/texts/{l}/en/"
        untar(rev_pair_dir + f'{l}-en.tgz', rev_pair_dir)

        src_xml = f"{pair_dir}/en-{l}/IWSLT17.TED.tst2017.mltlng.en-{l}.en.xml"
        tgt_xml = f"{rev_pair_dir}/{l}-en/IWSLT17.TED.tst2017.mltlng.{l}-en.{l}.xml"

        src_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.eng_Latn"
        tgt_output = f"{dataset_directory}/test.eng_Latn-{map_langs[l]}.{map_langs[l]}"

        xml2txt(src_xml,  src_output)
        xml2txt(tgt_xml,  tgt_output)

    shutil.rmtree(tmp_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to download individual public copora for NLLB"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        required=True,
        help="directory to save downloaded data",
    )
    # /private/home/elbayadm/data/iwslt_source
    parser.add_argument(
        "--pre-download",
        "-p",
        type=str,
        required=True,
        help="directory of pre-downloaded files",
    )
    args = parser.parse_args()

    directory = args.directory

    if not os.path.isdir(directory):
        print(f"Creating directory: {directory}")
        os.mkdir(directory)

    download_mafand(directory)
    download_wat(directory)
    download_floresv1(directory)
    download_wmt(directory)
    download_tico(directory)

    # Require pre-downloading data and agreeing to terms of use
    download_autshumato(directory, args.pre_download)
    download_madar(directory, args.pre_download)
    download_iwslt(directory, args.pre_download)

