class Dictionary(object):
    """A mapping from symbols to consecutive integers"""
    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}

    def __getitem__(self, i):
        return self.symbols[i]

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        return self.indices[sym]

    @staticmethod
    def load(f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """

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
