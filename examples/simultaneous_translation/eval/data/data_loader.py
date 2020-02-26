import os.path as op
import json
import soundfile as sf

class TextDataLoader(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.source = open(data_root)

    def load(self):
        for sent_id, line in enumerate(self.source):
            yield sent_id, line.strip().split()
        self.source.close()

class AudioDataLoader(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.source = open(data_root)

    def load(self):
        for sent_id, utterence in enumerate(json.load(self.source)["utts"].values()):
            wav_path = utterence["input"]["path"]
            assert op.isfile(wav_path) and wav_path.endswith('.wav')
            frames_10ms = 16000 // 100
            wav_blocks = [b.tolist() for b in sf.blocks(wav_path, blocksize=frames_10ms)]
            yield sent_id, wav_blocks
        self.source.close()
