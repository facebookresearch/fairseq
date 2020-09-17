import xml.etree.ElementTree as ET
import xmltodict

'''

In this file, all the xlm files of datasets will be cleaned up into plain text.
Then these plain texts will be saved with their document info.
The final output of this file will be text files(en/de) for plain sentences which are for training NMT models
and text files for sentences and their document info which are for next step(idk yet).

'''

tree = ET.parse('country_data.xml')

xml_path = "orig/EMEA_xml/de-en.xml"

xmlfile_dict = xmltodict.parse(xml_path)