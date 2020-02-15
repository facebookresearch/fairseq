import os.path as op
import json

class TextDataLoader(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.source = open(data_root)

    def load(self):
        for sent_id, line in enumerate(self.source):
            yield sent_id, line.strip().split()
        self.source.close()
