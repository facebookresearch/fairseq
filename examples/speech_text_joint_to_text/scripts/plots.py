import pickle
import sys
import os
import matplotlib.pylab as plt

#from scipy.ndimage import *
#import cv2
#
#def smooth(a):
#    return grey_opening(grey_closing((a+1) * 128, size=(3,3)), size=(3,3))
#
#def smooth_binary(a):
#    return binary_erosion(binary_closing(a))
#
#contours, _ = cv2.findContours(smooth_binary(s > 0.65).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#plt.imshow(smooth_binary(s > 0.65))
#for i, contour in enumerate(contours):
#    x, y, w, h = cv2.boundingRect(contour)
#    plt.gca().add_patch(Rectangle((x-1,y-1),w,h,linewidth=1,edgecolor='r',facecolor='none'))
#SIM_FILE = "symilarities.pickle"
#PHONEMES_FILE = "test_ep_ph.tsv"

SIM_FILE = sys.argv[1]
PHONEMES_FILE = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

with open(SIM_FILE, 'rb') as f:
    sm = pickle.load(f)

with open(PHONEMES_FILE, 'r') as f:
    phoenems = [l.strip().split('\t')[-1].split(' ') + ['<eos>'] for l in f]
    del phoenems[0]

os.mkdir(OUTPUT_DIR)

for key in sm:
    s = sm[key][: (sm[key][:, 0] != float('-inf')).sum(), : (sm[key][0] != float('-inf')).sum()].numpy()
    plt.figure(figsize=(s.shape[0]/2, s.shape[0]/2))
    heatmap = plt.imshow(s, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    heatmap.axes.xaxis.set_ticks(range(s.shape[1]))
    heatmap.axes.xaxis.set_ticklabels(phoenems[key])
    plt.savefig(f"{OUTPUT_DIR}/f{key}.png")
