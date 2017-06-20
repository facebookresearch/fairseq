class Dictionary(object):
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}

    def __getitem__(self, i):
        return self.symbols[i]

    def __len__(self):
        return len(self.symbols)

    def index(self, sym):
        return self.indices[sym]

    @staticmethod
    def load(f):
        if isinstance(f, str):
            with open(f, 'r') as fd:
                return Dictionary.load(fd)

        d = Dictionary()
        for line in f.readlines():
            idx = line.rfind(" ")
            word = line[:idx]
            count = int(line[idx+1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d
