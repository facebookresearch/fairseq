import numpy as np
import sys
import csv
csv.field_size_limit(sys.maxsize)


def text2arr(text_path):
    with open(text_path) as f:
        arr_from_text = [line.rstrip() for line in f]
    return arr_from_text


def arr2txt(arr, file_name):
    with open("custom_split_data/"+file_name+".txt" , "w") as txt_file:
        for line in arr:
            txt_file.write("".join(line) + "\n")


def split_arr(arr, n_train, n_valid, n_test):
    total_num = len(arr)
    if n_train+n_valid+n_test <= total_num:
        train_set, valid_set, test_set = arr[:n_train], arr[n_train:n_valid+n_train], \
                                         arr[n_valid+n_train:n_valid+n_train+n_test]
    return train_set, valid_set, test_set


def format_filename(lang, sets):
    return ".".join(map(str, [sets] + [lang]))


# EMEA path
emea_de_path = "custom_data/EMEA/EMEA.de-en.de"
emea_en_path = "custom_data/EMEA/EMEA.de-en.en"

# EMEA arrs - de, en
arr_emea_de = text2arr(emea_de_path)
arr_emea_en = text2arr(emea_en_path)
print(arr_emea_en[:100])
# split arrs into train, valid and test sets
train_de, valid_de, test_de = split_arr(arr_emea_de, 10000, 1000, 2000)
train_en, valid_en, test_en = split_arr(arr_emea_en, 10000, 1000, 2000)

#write to txt files!
names_de = ["train.de", "valid.de", "test.de"]
names_en = ["train.en", "valid.en", "test.en"]
langs = ["de", "en"]
sets = ["train", "valid", "test"]

arr2txt(train_de, "train.de")
arr2txt(valid_de, "valid.de")
arr2txt(test_de, "test.de")

arr2txt(train_en, "train.en")
arr2txt(valid_en, "valid.en")
arr2txt(test_en, "test.en")