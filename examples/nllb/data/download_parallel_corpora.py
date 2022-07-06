# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import gzip
import os
import re
import shutil
import tarfile
import zipfile

import openpyxl
import requests
from translate.storage.tmx import tmxfile

"""
Dependencies:
openpyxl (pip)
gsutil (pip)
translate-toolkit (pip)

See README in local directory for important notes on usage / data coverage
"""


def download_file(download_url, download_path):
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url}!")
        return False
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")
    return True


def gzip_extract_and_remove(zipped_path):
    assert zipped_path.endswith(".gz")
    unzipped_path = zipped_path[:-3]

    with gzip.open(zipped_path, "rb") as f:
        file_content = f.read()
    with open(unzipped_path, "wb") as f:
        f.write(file_content)
    os.remove(zipped_path)
    print(f"Extracted and removed: {zipped_path}")
    return unzipped_path


def concat_url_text_contents_to_file(url_list, download_path):
    with open(download_path, "w") as f:
        for download_url in url_list:
            response = requests.get(download_url)
            if not response.ok:
                print(f"Could not download from {download_url}!")
                return False
            f.write(response.text)
            if not response.text.endswith("\n"):
                f.write("\n")
    print(f"Wrote: {download_path}")
    return True


def download_TIL(directory):
    """
    https://arxiv.org/pdf/2109.04593.pdf
    https://github.com/turkic-interlingua/til-mt
    Total download: 22.4 GB
    """
    dataset_directory = os.path.join(directory, "til")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving TIL data to:", dataset_directory)

    til_train_url = "gs://til-corpus/corpus/train/*"
    command = f"gsutil -m cp -r {til_train_url} {dataset_directory}"
    print(f"Running: {command}")
    try:
        os.system(command)
    except Exception as e:
        print(f"gsutil download failed! \n{e}")

    # we map to three-letter ISO 639-3 code (for NLLB langs)
    lang_map = {
        "az": "aze",
        "ba": "bak",
        "cv": "chv",
        "en": "eng",
        "kk": "kaz",
        "ky": "kir",
        "ru": "rus",
        "tk": "tuk",
        "tr": "tur",
        "tt": "tat",
        "ug": "uig",
        "uz": "uzb",
    }

    pair_directories = os.listdir(dataset_directory)
    for pair_directory in pair_directories:
        try:
            src, tgt = pair_directory.split("-")
        except:
            print("Unexpected TIL pair directory name:", pair_directory)

        try:
            dest_src = lang_map[src]
        except:
            dest_src = src

        try:
            dest_tgt = lang_map[tgt]
        except:
            dest_tgt = tgt

        pair_directory_path = os.path.join(dataset_directory, pair_directory)
        os.rename(
            os.path.join(pair_directory_path, f"{src}-{tgt}.{src}"),
            os.path.join(pair_directory_path, f"til.{dest_src}"),
        )
        os.rename(
            os.path.join(pair_directory_path, f"{src}-{tgt}.{tgt}"),
            os.path.join(pair_directory_path, f"til.{dest_tgt}"),
        )
        renamed_pair_directory = os.path.join(
            dataset_directory, f"{dest_src}-{dest_tgt}"
        )
        os.rename(pair_directory_path, renamed_pair_directory)
        print(f"Renamed to: {renamed_pair_directory}")


def download_TICO(directory, verbose=False):
    """
    https://tico-19.github.io/
    Total after extraction: 130M
    """
    dataset_directory = os.path.join(directory, "tico")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving TICO data to:", dataset_directory)

    source_langs = {
        "am": "amh",
        "ar": "ara",
        "en": "eng",
        "es-LA": "spa",
        "fa": "fas",
        "fr": "fra",
        "hi": "hin",
        "id": "ind",
        "ku": "kmr",
        "pt-BR": "por",
        "ru": "rus",
        "zh": "zho",
    }
    target_langs = {
        "am": "amh",
        "ar": "ara",
        "bn": "ben",
        "ckb": "ckb",
        "din": "din",
        "es-LA": "spa",
        "fa": "fas",
        "fr": "fra",
        "fuv": "fuv",
        "ha": "hau",
        "hi": "hin",
        "id": "ind",
        "km": "khm",
        "kr": "knc",
        "ku": "kmr",
        "lg": "lug",
        "ln": "lin",
        "mr": "mar",
        "ms": "msa",
        "my": "mya",
        "ne": "npi",
        "nus": "nus",
        "om": "orm",
        "prs": "prs",
        "pt-BR": "por",
        "ps": "pus",
        "ru": "rus",
        "rw": "kin",
        "so": "som",
        "sw": "swh",
        "ta": "tam",
        "ti_ET": "tir_ET",  # we combine both Tigrinya varieties
        "ti_ER": "tir_ER",  # we combine both Tigrinya varieties
        "tl": "tgl",
        "ur": "urd",
        "zh": "zho",
        "zu": "zul",
    }

    for source in source_langs:
        for target in target_langs:
            url = f"https://tico-19.github.io/data/TM/all.{source}-{target}.tmx.zip"
            response = requests.get(url)
            if not response.ok:
                if verbose:
                    print("Could not download data for {source}-{target}! Skipping...")
                continue

            lang1 = source_langs[source]
            lang2 = target_langs[target]
            if lang2 < lang1:
                lang1, lang2 = lang2, lang1
            direction_directory = os.path.join(dataset_directory, f"{lang1}-{lang2}")
            os.makedirs(direction_directory, exist_ok=True)

            download_path = os.path.join(
                direction_directory, f"all.{source}-{target}.tmx.zip"
            )
            open(download_path, "wb").write(response.content)
            print(f"Wrote: {download_path}")

            with zipfile.ZipFile(download_path, "r") as z:
                z.extractall(direction_directory)
            tmx_file_path = os.path.join(
                direction_directory, f"all.{source}-{target}.tmx"
            )
            os.remove(download_path)
            print(f"Extracted and removed: {download_path}")

            # extract source and target from tmx format
            # use 3-letter codes
            src = source_langs[source]
            tgt = target_langs[target]

            with open(tmx_file_path, "rb") as f:
                tmx_data = tmxfile(f, source, target)
            source_path = os.path.join(
                direction_directory, f"tico19.{lang1}-{lang2}.{src}"
            )
            with open(source_path, "w") as f:
                for node in tmx_data.unit_iter():
                    f.write(node.source)
                    f.write("\n")
            print(f"Wrote: {source_path}")

            target_path = os.path.join(
                direction_directory, f"tico19.{lang1}-{lang2}.{tgt}"
            )
            with open(target_path, "w") as f:
                for node in tmx_data.unit_iter():
                    f.write(node.target)
                    f.write("\n")
            print(f"Wrote: {target_path}")
            os.remove(tmx_file_path)
            print(f"Deleted: {tmx_file_path}")


def download_IndicNLP(directory):
    """
    http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/
    Total after extraction: 3.1GB
    """
    dataset_directory = os.path.join(directory, "indic_nlp")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Indic NLP data to:", dataset_directory)

    download_url = (
        "http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/indic_wat_2021.tar.gz"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for Indic NLP!")
        return
    download_path = os.path.join(dataset_directory, "indic_wat_2021.tar.gz")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    with tarfile.open(download_path) as tar:
        tar.extractall(dataset_directory)
    os.remove(download_path)
    print(f"Deleted: {download_path}")
    print(f"Extracted to: {dataset_directory}")


def download_Lingala_Song_Lyrics(directory):
    """
    https://github.com/espoirMur/songs_lyrics_webscrap
    """
    dataset_directory = os.path.join(directory, "lingala_songs")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Lingala song lyric data to:", dataset_directory)

    download_url = "https://raw.githubusercontent.com/espoirMur/songs_lyrics_webscrap/master/data/all_data.csv"
    response = requests.get(download_url)
    if not response.ok:
        print(
            f"Could not download from {download_url} ... aborting for Lingala song lyrics!"
        )
        return
    download_path = os.path.join(dataset_directory, "all_data.csv")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    content_lines = open(download_path).readlines()
    fr_examples = []
    lin_examples = []
    for pair_line in content_lines[1:]:  # first line specifies languages
        fr, lin = pair_line.split("|")

        # multiple spaces separate song lines within stanzas
        fr_lines = re.sub("\s\s+", "\t", fr).split("\t")
        lin_lines = re.sub("\s\s+", "\t", lin).split("\t")

        if len(fr_lines) == len(lin_lines):
            fr_examples.extend(fr_lines)
            lin_examples.extend(lin_lines)
        else:
            fr_examples.append(" ".join(fr_lines))
            lin_examples.append(" ".join(lin_lines))
    fr_examples = [examp.strip() for examp in fr_examples]
    lin_examples = [examp.strip() for examp in lin_examples]

    fr_file = os.path.join(dataset_directory, "songs.fr-lin.fr")
    with open(fr_file, "w") as f:
        f.write("\n".join(fr_examples))
    print(f"Wrote: {fr_file}")

    lin_file = os.path.join(dataset_directory, "songs.fr-lin.lin")
    with open(lin_file, "w") as f:
        f.write("\n".join(lin_examples))
    print(f"Wrote: {lin_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_FFR(directory):
    """
    https://arxiv.org/abs/2006.09217
    https://github.com/bonaventuredossou/ffr-v1/tree/master/FFR-Dataset
    """
    dataset_directory = os.path.join(directory, "ffr")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving FFR (Fon-French) data to:", dataset_directory)

    download_url = "https://raw.githubusercontent.com/bonaventuredossou/ffr-v1/master/FFR-Dataset/FFR%20Dataset%20v2/ffr_dataset_v2.txt"
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for FFR!")
        return
    download_path = os.path.join(dataset_directory, "ffr_dataset_v2.txt")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    fon_filename = os.path.join(dataset_directory, "ffr.fon-fra.fon")
    fra_filename = os.path.join(dataset_directory, "ffr.fon-fra.fra")

    with open(download_path) as f, open(fon_filename, "w") as fon, open(
        fra_filename, "w"
    ) as fra:
        for joint_line in f:
            # one line has a tab in the French side: "A tout seigneur \t tout honneur"
            fon_line, fra_line = joint_line.split("\t", 1)
            fon.write(fon_line.strip() + "\n")
            fra.write(fra_line.strip() + "\n")
    print(f"Wrote: {fon_filename}")
    print(f"Wrote: {fra_filename}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_Mburisano_Covid(directory):
    """
    https://repo.sadilar.org/handle/20.500.12185/536
    """
    print("IMPORTANT!")
    print("By downloading this corpus, you agree to the terms of use found here:")
    print("https://sadilar.org/index.php/en/guidelines-standards/terms-of-use")

    dataset_directory = os.path.join(directory, "mburisano")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Mburisano data to:", dataset_directory)

    download_url = "https://repo.sadilar.org/bitstream/20.500.12185/536/1/mburisano_multilingual_corpus.csv"
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for Mburisano!")
        return
    download_path = os.path.join(dataset_directory, "mburisano_multilingual_corpus.csv")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    # line 0 contains language names
    csv_lines = open(download_path).readlines()[1:]
    data = {
        "afr": [],  # Afrikaans
        "eng": [],  # English
        "nde": [],  # isiNdebele
        "sot": [],  # Sesotho
        "ssw": [],  # Siswati
        "tsn": [],  # Setswana
        "tso": [],  # Xitsonga
        "ven": [],  # Tshiven·∏ìa
        "xho": [],  # isiXhosa
        "zul": [],  # isiZulu
    }
    for line in csv.reader(csv_lines):
        if len(line) == 11:
            afr, eng, nde, xho, zul, sot, _, tsn, ssw, ven, tso = line
        else:
            print("Does not have correct number of fields!")
            print(line)
        data["afr"].append(afr)
        data["eng"].append(eng)
        data["nde"].append(nde)
        data["sot"].append(sot)
        data["ssw"].append(ssw)
        data["tsn"].append(tsn)
        data["tso"].append(tso)
        data["ven"].append(ven)
        data["xho"].append(xho)
        data["zul"].append(zul)

    for source, target in [
        ("afr", "eng"),
        ("eng", "nde"),
        ("eng", "sot"),
        ("eng", "ssw"),
        ("eng", "tsn"),
        ("eng", "tso"),
        ("eng", "ven"),
        ("eng", "xho"),
        ("eng", "zul"),
    ]:
        direction_directory = os.path.join(dataset_directory, f"{source}-{target}")
        os.makedirs(direction_directory, exist_ok=True)

        source_file = os.path.join(direction_directory, f"mburisano.{source}")
        with open(source_file, "w") as f:
            for line in data[source]:
                f.write(line)
                f.write("\n")
        print(f"Wrote: {source_file}")

        target_file = os.path.join(direction_directory, f"mburisano.{target}")
        with open(target_file, "w") as f:
            for line in data[target]:
                f.write(line)
                f.write("\n")
        print(f"Wrote: {target_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_XhosaNavy(directory):
    """
    https://opus.nlpl.eu/XhosaNavy.php
    """
    dataset_directory = os.path.join(directory, "xhosa_navy")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Xhosa Navy data to:", dataset_directory)

    download_url = (
        "https://opus.nlpl.eu/download.php?f=XhosaNavy/v1/moses/en-xh.txt.zip"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for Xhosa Navy!")
        return
    download_path = os.path.join(dataset_directory, "en-xh.txt.zip")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    extract_directory = os.path.join(dataset_directory, "extract")
    os.makedirs(extract_directory, exist_ok=True)
    with zipfile.ZipFile(download_path, "r") as z:
        z.extractall(extract_directory)
    print("Extracted to:", extract_directory)

    eng_file = os.path.join(dataset_directory, "XhosaNavy.eng")
    os.rename(
        os.path.join(extract_directory, "XhosaNavy.en-xh.en"),
        eng_file,
    )
    print(f"Wrote: {eng_file}")

    xho_file = os.path.join(dataset_directory, "XhosaNavy.xho")
    os.rename(
        os.path.join(extract_directory, "XhosaNavy.en-xh.xh"),
        xho_file,
    )
    print(f"Wrote: {xho_file}")

    shutil.rmtree(extract_directory)
    print(f"Deleted tree: {extract_directory}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_Menyo20K(directory):
    """
    https://arxiv.org/abs/2103.08647
    https://github.com/uds-lsv/menyo-20k_MT
    """
    dataset_directory = os.path.join(directory, "menyo20k")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving MENYO-20k data to:", dataset_directory)

    download_url = (
        "https://raw.githubusercontent.com/uds-lsv/menyo-20k_MT/master/data/train.tsv"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for MENYO-20k!")
        return
    download_path = os.path.join(dataset_directory, "train.tsv")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    # line 0 contains language names
    tsv_lines = open(download_path).readlines()[1:]
    source_file = os.path.join(dataset_directory, "menyo20k.eng")
    target_file = os.path.join(dataset_directory, "menyo20k.yor")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for line in csv.reader(tsv_lines, delimiter="\t"):
            source_line, target_line = line
            src.write(source_line.strip())
            src.write("\n")
            tgt.write(target_line.strip())
            tgt.write("\n")

    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")


def download_FonFrench(directory):
    """
    https://zenodo.org/record/4266935#.YaTu0fHMJDY
    """
    dataset_directory = os.path.join(directory, "french_fongbe")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving French-Fongbe data to:", dataset_directory)

    download_url = (
        "https://zenodo.org/record/4266935/files/French_to_fongbe.csv?download=1"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for French-Fongbe!")
        return
    download_path = os.path.join(dataset_directory, "French_to_fongbe.csv")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    # line 0 contains language names
    csv_lines = open(download_path).readlines()[1:]
    source_file = os.path.join(dataset_directory, "french_fongbe.fon")
    target_file = os.path.join(dataset_directory, "french_fongbe.fra")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for line in csv.reader(csv_lines):
            source_line, target_line = line
            src.write(source_line.strip())
            src.write("\n")
            tgt.write(target_line.strip())
            tgt.write("\n")
    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_FrenchEwe(directory):
    """
    https://zenodo.org/record/4266935#.YaTu0fHMJDY
    """
    dataset_directory = os.path.join(directory, "french_ewe")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving French-Ewe data to:", dataset_directory)

    download_url = (
        "https://zenodo.org/record/4266935/files/French_to_ewe_dataset.xlsx?download=1"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for French-Ewe!")
        return
    download_path = os.path.join(dataset_directory, "French_to_ewe_dataset.xlsx")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    wb = openpyxl.load_workbook(download_path)
    french_sheet = wb["French"]
    ewe_sheet = wb["Ewe"]

    french_examples = []
    ewe_examples = []
    for french_row, ewe_row in zip(french_sheet.rows, ewe_sheet.rows):
        if french_row[1].value is None or ewe_row[1].value is None:
            continue
        # preserve file alignment by removing newlines
        french_sent = french_row[1].value.strip().replace("\n", " ")
        ewe_sent = ewe_row[1].value.strip().replace("\n", " ")
        french_examples.append(french_sent)
        ewe_examples.append(ewe_sent)

    source_file = os.path.join(dataset_directory, "french_ewe.fra")
    target_file = os.path.join(dataset_directory, "french_ewe.ewe")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for fra, ewe in zip(french_examples, ewe_examples):
            src.write(fra)
            src.write("\n")
            tgt.write(ewe)
            tgt.write("\n")
    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_Akuapem(directory):
    """
    https://arxiv.org/pdf/2103.15625.pdf
    """
    dataset_directory = os.path.join(directory, "akuapem")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Akuapem data (English - Akuapem Twi):", dataset_directory)

    download_url = (
        "https://zenodo.org/record/4432117/files/verified_data.csv?download=1"
    )
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting for Akuapem!")
        return
    download_path = os.path.join(dataset_directory, "verified_data.csv")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    # line 0 contains language names
    csv_lines = open(download_path).readlines()[1:]
    source_file = os.path.join(dataset_directory, "akuapem.eng")
    target_file = os.path.join(dataset_directory, "akuapem.aka")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for line in csv.reader(csv_lines):
            source_line, target_line = line
            src.write(source_line.strip())
            src.write("\n")
            tgt.write(target_line.strip())
            tgt.write("\n")
    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_GELR(directory):
    """
    https://www.ijert.org/research/gelr-a-bilingual-ewe-english-corpus-building-and-evaluation-IJERTV10IS080214.pdf
    """
    pass


def download_GiossaMedia(directory):
    """
    https://github.com/sgongora27/giossa-gongora-guarani-2021
    """
    dataset_directory = os.path.join(directory, "giossa")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Giossa data (Spanish - Guarani):", dataset_directory)

    guarani_examples = []
    spanish_examples = []

    for name, download_url in [
        (
            "parallel_march",
            "https://github.com/sgongora27/giossa-gongora-guarani-2021/blob/main/ParallelSet/parallel_march.zip?raw=true",
        ),
        (
            "parallel_april",
            "https://github.com/sgongora27/giossa-gongora-guarani-2021/blob/main/ParallelSet/parallel_april.zip?raw=true",
        ),
    ]:
        response = requests.get(download_url)
        if not response.ok:
            print(f"Could not download from {download_url} ... aborting!")
            return
        download_path = os.path.join(dataset_directory, f"{name}.zip")
        open(download_path, "wb").write(response.content)
        print(f"Wrote: {download_path}")

        with zipfile.ZipFile(download_path, "r") as z:
            z.extractall(dataset_directory)
        subset_directory = os.path.join(dataset_directory, name)
        print(f"Extracted to: {subset_directory}")
        os.remove(download_path)
        print(f"Deleted: {download_path}")

        for filename in os.listdir(subset_directory):
            aligned_file = os.path.join(subset_directory, filename)
            with open(aligned_file) as f:
                contents = f.read()

            aligned_pairs = contents.split("gn: ")
            for pair in aligned_pairs:
                if len(pair.strip()) == 0:
                    continue
                try:
                    grn, spa = pair.split("\nes: ")
                    grn = grn.strip().replace("\n", " ")
                    spa = spa.strip().replace("\n", " ")
                except:
                    print(f"Expected pair separated by 'es: ' but got: {pair}!")
                    import pdb

                    pdb.set_trace()
                # begins with "gn: "
                grn = grn[4:]
                guarani_examples.append(grn)
                spanish_examples.append(spa)
        shutil.rmtree(subset_directory)
        print(f"Deleted tree: {subset_directory}")

    guarani_file = os.path.join(dataset_directory, "giossa.grn")
    with open(guarani_file, "w") as f:
        for sent in guarani_examples:
            f.write(sent)
            f.write("\n")

    spanish_file = os.path.join(dataset_directory, "giossa.spa")
    with open(spanish_file, "w") as f:
        for sent in spanish_examples:
            f.write(sent)
            f.write("\n")


def download_KinyaSMT(directory):
    """
    https://github.com/pniyongabo/kinyarwandaSMT
    """
    dataset_directory = os.path.join(directory, "kinya_smt")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Kinya-SMT:", dataset_directory)

    kin_examples = []
    eng_examples = []

    download_url = "https://github.com/sgongora27/giossa-gongora-guarani-2021/blob/main/ParallelSet/parallel_march.zip?raw=true"

    for name, download_url in [
        (
            "bible.en",
            "https://raw.githubusercontent.com/pniyongabo/kinyarwandaSMT/master/train-data/bible.en",
        ),
        (
            "bible.kn",
            "https://raw.githubusercontent.com/pniyongabo/kinyarwandaSMT/master/train-data/bible.kn",
        ),
        (
            "train.en",
            "https://raw.githubusercontent.com/pniyongabo/kinyarwandaSMT/master/train-data/train.en",
        ),
        (
            "train.kn",
            "https://raw.githubusercontent.com/pniyongabo/kinyarwandaSMT/master/train-data/train.kn",
        ),
    ]:
        response = requests.get(download_url)
        if not response.ok:
            print(f"Could not download from {download_url} ... aborting!")
            return
        download_path = os.path.join(dataset_directory, f"{name}.zip")
        open(download_path, "wb").write(response.content)
        print(f"Wrote: {download_path}")

        with open(download_path) as f:
            if name.endswith("kn"):
                kin_examples.extend(f.readlines())
            else:
                eng_examples.extend(f.readlines())
        os.remove(download_path)

    assert len(kin_examples) == len(eng_examples)
    output_path = os.path.join(dataset_directory, "kinyasmt.kin")
    with open(output_path, "w") as f:
        for sent in kin_examples:
            f.write(sent.strip())
            f.write("\n")
    print(f"Wrote: {output_path}")

    output_path = os.path.join(dataset_directory, "kinyasmt.eng")
    with open(output_path, "w") as f:
        for sent in eng_examples:
            f.write(sent.strip())
            f.write("\n")
    print(f"Wrote: {output_path}")


def download_translation_memories_from_Nynorsk(directory):
    """
    (including Nynorsk)
    https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-47/
    """
    dataset_directory = os.path.join(directory, "nynorsk_memories")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Translation Memorie from Nynorsk:", dataset_directory)

    download_url = "https://www.nb.no/sbfil/tekst/2011_2019_tm_npk_ntb.tar.gz"
    response = requests.get(download_url)
    if not response.ok:
        print(f"Could not download from {download_url} ... aborting!")
        return
    download_path = os.path.join(dataset_directory, f"011_2019_tm_npk_ntb.tar.gz")
    open(download_path, "wb").write(response.content)
    print(f"Wrote: {download_path}")

    with tarfile.open(download_path) as tar:
        tar.extractall(dataset_directory)
    os.remove(download_path)
    tmx_file_path = os.path.join(dataset_directory, f"2011_2019_tm_npk_ntb.tmx")

    with open(tmx_file_path, "rb") as f:
        # Norwegian Bokmal to Norwegian Nynorsk
        tmx_data = tmxfile(f, "NB", "NN")
    os.remove(tmx_file_path)

    # Norwegian Bokmål
    source_path = os.path.join(dataset_directory, f"nynorsk_memories.nob")
    with open(source_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.source)
            f.write("\n")
    print(f"Wrote: {source_path}")

    # Norwegian Nynorsk
    target_path = os.path.join(dataset_directory, f"nynorsk_memories.nno")
    with open(target_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.target)
            f.write("\n")
    print(f"Wrote: {target_path}")


def download_mukiibi(directory):
    """
    https://zenodo.org/record/5089560#.YaipovHMJDZ
    """
    dataset_directory = os.path.join(directory, "mukiibi")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Makerere MT (English - Luganda) data:", dataset_directory)

    download_url = (
        "https://zenodo.org/record/5089560/files/English-Luganda.tsv?download=1"
    )
    download_path = os.path.join(dataset_directory, "English-Luganda.tsv")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Makerere MT (English - Luganda)!")
        return

    # line 0 contains language names
    tsv_lines = open(download_path).readlines()[1:]
    source_file = os.path.join(dataset_directory, "mukiibi.eng")
    target_file = os.path.join(dataset_directory, "mukiibi.lug")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for line in csv.reader(tsv_lines, delimiter="\t"):
            # empty third "column" (line-ending tab)
            source_line, target_line, _ = line
            src.write(source_line.strip())
            src.write("\n")
            tgt.write(target_line.strip())
            tgt.write("\n")
    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")
    os.remove(download_path)
    print(f"Deleted: {download_path}")


def download_umsuka(directory):
    """
    https://zenodo.org/record/5035171#.YaippvHMJDZ
    """
    dataset_directory = os.path.join(directory, "umsuka")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Umsuka (isiZulu - English) data:", dataset_directory)

    eng_examples = []
    zul_examples = []

    download_url = (
        "https://zenodo.org/record/5035171/files/en-zu.training.csv?download=1"
    )
    download_path = os.path.join(dataset_directory, "en-zu.training.csv")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Umsuka (isiZulu - English)!")
        return

    # line 0 contains columm names
    with open(download_path) as f:
        csv_lines = f.readlines()[1:]
    for line in csv.reader(csv_lines):
        # third column contains data source
        source_line, target_line, _ = line
        eng_examples.append(source_line.strip().replace("\n", " "))
        zul_examples.append(target_line.strip().replace("\n", " "))
    os.remove(download_path)
    print(f"Deleted: {download_path}")

    source_file = os.path.join(dataset_directory, "umsuka.eng")
    target_file = os.path.join(dataset_directory, "umsuka.zul")
    with open(source_file, "w") as src, open(target_file, "w") as tgt:
        for eng, zul in zip(eng_examples, zul_examples):
            src.write(eng)
            src.write("\n")
            tgt.write(zul)
            tgt.write("\n")
    print(f"Wrote: {source_file}")
    print(f"Wrote: {target_file}")


def download_CMU_Haitian_Creole(directory):
    """
    http://www.speech.cs.cmu.edu/haitian/text/
    """
    dataset_directory = os.path.join(directory, "cmu_hatian")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving CMU Hatian Creole data:", dataset_directory)

    download_url = (
        "http://www.speech.cs.cmu.edu/haitian/text/1600_medical_domain_sentences.en"
    )
    download_path = os.path.join(dataset_directory, "cmu.eng")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for CMU Hatian Creole!")
        return

    download_url = (
        "http://www.speech.cs.cmu.edu/haitian/text/1600_medical_domain_sentences.ht"
    )
    download_path = os.path.join(dataset_directory, "cmu.hat")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for CMU Hatian Creole!")
        return


def download_Bianet(directory):
    """
    https://opus.nlpl.eu/Bianet.php
    Ataman, D. (2018) Bianet: A Parallel News Corpus in Turkish, Kurdish and English. In Proceedings of the LREC 2018 Workshop MLP-Moment. pp. 14-17
    """
    dataset_directory = os.path.join(directory, "bianet")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Bianet data:", dataset_directory)

    # eng-kur
    download_url = "https://opus.nlpl.eu/download.php?f=Bianet/v1/tmx/en-ku.tmx.gz"
    download_path = os.path.join(dataset_directory, "en-ku.tmx.gz")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Bianet!")
        return
    tmx_file_path = gzip_extract_and_remove(download_path)
    with open(tmx_file_path, "rb") as f:
        # English to Kurdish
        tmx_data = tmxfile(f, "en", "ku")
    os.remove(tmx_file_path)

    direction_directory = os.path.join(dataset_directory, "eng-kur")
    os.makedirs(direction_directory, exist_ok=True)

    source_path = os.path.join(direction_directory, f"bianet.eng")
    with open(source_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.source)
            f.write("\n")
    print(f"Wrote: {source_path}")

    target_path = os.path.join(direction_directory, f"bianet.kur")
    with open(target_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.target)
            f.write("\n")
    print(f"Wrote: {target_path}")

    # kur-tur
    download_url = "https://opus.nlpl.eu/download.php?f=Bianet/v1/tmx/ku-tr.tmx.gz"
    download_path = os.path.join(dataset_directory, "ku-tr.tmx.gz")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Bianet!")
        return
    tmx_file_path = gzip_extract_and_remove(download_path)
    with open(tmx_file_path, "rb") as f:
        # English to Kurdish
        tmx_data = tmxfile(f, "ku", "tr")
    os.remove(tmx_file_path)

    direction_directory = os.path.join(dataset_directory, "kur-tur")
    os.makedirs(direction_directory, exist_ok=True)

    source_path = os.path.join(direction_directory, f"bianet.kur")
    with open(source_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.source)
            f.write("\n")
    print(f"Wrote: {source_path}")

    target_path = os.path.join(direction_directory, f"bianet.tur")
    with open(target_path, "w") as f:
        for node in tmx_data.unit_iter():
            f.write(node.target)
            f.write("\n")
    print(f"Wrote: {target_path}")


def download_HornMT(directory):
    """
    https://github.com/asmelashteka/HornMT
    """
    dataset_directory = os.path.join(directory, "hornmt")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving HornMT data:", dataset_directory)

    lang_files = {}
    for lang in ("aar", "amh", "eng", "orm", "som", "tir"):
        download_url = f"https://raw.githubusercontent.com/asmelashteka/HornMT/main/data/{lang}.txt"
        download_path = os.path.join(dataset_directory, f"{lang}.txt")
        ok = download_file(download_url, download_path)
        if not ok:
            print("Aborting for HornMT!")
            return
        lang_files[lang] = download_path

    for source, target in [
        ("aar", "amh"),
        ("aar", "eng"),
        ("aar", "orm"),
        ("aar", "som"),
        ("aar", "tir"),
        ("amh", "eng"),
        ("amh", "orm"),
        ("amh", "som"),
        ("amh", "tir"),
        ("eng", "orm"),
        ("eng", "som"),
        ("eng", "tir"),
        ("orm", "som"),
        ("orm", "tir"),
        ("som", "tir"),
    ]:
        direction_directory = os.path.join(dataset_directory, f"{source}-{target}")
        os.makedirs(direction_directory, exist_ok=True)

        source_path = os.path.join(direction_directory, f"hornmt.{source}")
        shutil.copyfile(lang_files[source], source_path)
        print(f"Wrote: {source_path}")

        target_path = os.path.join(direction_directory, f"hornmt.{target}")
        shutil.copyfile(lang_files[target], target_path)
        print(f"Wrote: {target_path}")

    for filename in lang_files.values():
        os.remove(filename)


def download_minangNLP(directory):
    """
    https://github.com/fajri91/minangNLP
    """
    dataset_directory = os.path.join(directory, "minangnlp")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving Minang NLP data:", dataset_directory)

    direction_directory = os.path.join(dataset_directory, "min_Latn-ind")
    os.makedirs(direction_directory, exist_ok=True)

    # min_Latn
    download_url = "https://raw.githubusercontent.com/fajri91/minangNLP/master/translation/wiki_data/src_train.txt"
    download_path = os.path.join(direction_directory, f"minangnlp.min_Latn")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Minang NLP!")
        return

    # ind
    download_url = "https://raw.githubusercontent.com/fajri91/minangNLP/master/translation/wiki_data/tgt_train.txt"
    download_path = os.path.join(direction_directory, f"minangnlp.ind")
    ok = download_file(download_url, download_path)
    if not ok:
        print("Aborting for Minang NLP!")
        return


def download_aau(directory):
    """
    https://github.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages
    """
    dataset_directory = os.path.join(directory, "aau")
    os.makedirs(dataset_directory, exist_ok=True)
    print("Saving AAU data:", dataset_directory)

    # amh-eng
    direction_directory = os.path.join(dataset_directory, "amh-eng")
    os.makedirs(direction_directory, exist_ok=True)

    download_path = os.path.join(direction_directory, f"aau.amh")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/History/amh_eng/p_amh_ea",
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/amh_eng/p_amh_ea.txt",
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/News/amh_eng/amh_ea.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return

    download_path = os.path.join(direction_directory, f"aau.eng")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/History/amh_eng/p_eng_ea",
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/amh_eng/p_eng_ea.txt",
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/News/amh_eng/eng_ea.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return

    # eng-orm
    direction_directory = os.path.join(dataset_directory, "eng-orm")
    os.makedirs(direction_directory, exist_ok=True)

    download_path = os.path.join(direction_directory, f"aau.eng")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/oro_eng/p_eng_eo.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return

    download_path = os.path.join(direction_directory, f"aau.orm")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/oro_eng/p_oro_eo.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return

    # eng-tir
    direction_directory = os.path.join(dataset_directory, "eng-tir")
    os.makedirs(direction_directory, exist_ok=True)

    download_path = os.path.join(direction_directory, f"aau.eng")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/tig_eng/p_eng_et.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return

    download_path = os.path.join(direction_directory, f"aau.tir")
    ok = concat_url_text_contents_to_file(
        [
            "https://raw.githubusercontent.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages/master/Exp%20I-English%20to%20Local%20Lang/Legal/tig_eng/p_tig_et.txt",
        ],
        download_path,
    )
    if not ok:
        print("Aborting for AAU!")
        return


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
    args = parser.parse_args()

    directory = args.directory

    if not os.path.isdir(directory):
        print(f"Creating directory: {directory}")
        os.mkdir(directory)

    # Important:
    # By uncommenting the function below and downloading the
    # Mburisano_Covid corpus, you agree to the terms of use found here:
    # https://sadilar.org/index.php/en/guidelines-standards/terms-of-use
    # download_Mburisano_Covid(directory)

    download_TIL(directory)
    download_TICO(directory)
    download_IndicNLP(directory)
    download_Lingala_Song_Lyrics(directory)
    download_FFR(directory)
    download_Mburisano_Covid(directory)
    download_XhosaNavy(directory)
    download_Menyo20K(directory)
    download_FonFrench(directory)
    download_FrenchEwe(directory)
    download_Akuapem(directory)
    download_GiossaMedia(directory)
    download_KinyaSMT(directory)
    download_translation_memories_from_Nynorsk(directory)
    download_mukiibi(directory)
    download_umsuka(directory)
    download_CMU_Haitian_Creole(directory)
    download_Bianet(directory)
    download_HornMT(directory)
    download_minangNLP(directory)
    download_aau(directory)
