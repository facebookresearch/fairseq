import xml.etree.ElementTree as ET
from typing import Dict
import xmltodict
import pandas as pd
import sys
import csv

'''
In this file, all the xlm files of datasets will be cleaned up into plain text.
Then these plain texts will be saved with their document info.
The final output of this file will be text files(en/de) for plain sentences which are for training NMT models
and text files for sentences and their document info which are for next step(idk yet).
'''

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


# xmlfile_dict = xmltodict.parse(xml_path)
# print(xmlfile_dict)

def xml2dict(xml_path):
    with open(xml_path) as fd:
        orig_align_dict= xmltodict.parse(fd.read())
    return orig_align_dict


# def split_dict2xml(orig_align_dict):
    # iterate all the dict and split based on their docs
    # then I will need a function to write this dict(or maybe array) to xml file :D
    # the xml files will have numbers(from 0 till ..) this is just for make it easy for me
    # then finally, I will have splited alignment info xml files.
    # OPUS tool will read this file and clean up/ make the ready datasets



xml_path = "orig/EMEA_orig/de-en.xml"
with open(xml_path) as fd:
    orig_align_dict = xmltodict.parse(fd.read(), process_namespaces=True)
print(orig_align_dict['cesAlign']['linkGrp'][1])    # just check how it looks like
example = orig_align_dict['cesAlign']['linkGrp'][0]


# make a 'root' to the splitted xml files - because it complains that there is no 'root'.
example = {'linkGrp': example}
out = xmltodict.unparse(example, pretty=True)
with open("orig/EMEA_xml/xces_files/example.xml", 'a') as file:
    file.write(out)
# with open("orig/EMEA_xml/xces_files/example.xml", 'a', encoding="utf16", errors='ignore') as file:
#     file.write(out)

'''
for file in orig_align_dict['cesAlign']['linkGrp']:
    print('=================================================== /n')
    print(file)

########################################3
xml = open(path, "r")
org_xml = xml.read()
dict_xml = xmltodict.parse(org_xml, process_namespaces=True)

out = xmltodict.unparse(dict_xml, pretty=True)
with open("path/new.qml", 'a') as file:
    file.write(out.encode('utf-8'))
'''
