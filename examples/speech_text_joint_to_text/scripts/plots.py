import pickle
import sys
import os
import matplotlib.pylab as plt


with open(sys.argv[1], 'rb') as f:
    sm = pickle.load(f)

with open(sys.argv[2], 'r') as f:
    phoenems = [l.strip().split('\t')[-1].split(' ') + ['<eos>'] for l in f]
    del phoenems[0]

os.mkdir(sys.argv[3])

for key in sm:
    s = sm[key][: (sm[key][:, 0] != float('-inf')).sum(), : (sm[key][0] != float('-inf')).sum()].numpy()
    plt.figure(figsize=(s.shape[0], s.shape[0]))
    heatmap = plt.imshow(s, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    heatmap.axes.xaxis.set_ticks(range(s.shape[1]))
    heatmap.axes.xaxis.set_ticklabels(phoenems[key])
    plt.savefig(f"{sys.argv[3]}/f{key}.png")
