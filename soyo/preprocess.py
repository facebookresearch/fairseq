import xml.etree.ElementTree as ET
from typing import Dict
import xmltodict
import pandas as pd
import sys
import csv


csv.field_size_limit(sys.maxsize)


# read file and save it as array
def text2arr(text_path):
    with open(text_path) as f:
        arr_from_text = [line.rstrip() for line in f]
    return arr_from_text


def arr2txt(arr, file_name):
    with open("custom_split_data/"+file_name, "w") as txt_file:
        for line in arr:
            txt_file.write("".join(line) + "\n")


# 