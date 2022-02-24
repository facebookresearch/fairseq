import h5py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# [0, 488766], [488766, 513766], [513766, 538766]

# # plot test
# h5 = h5py.File('/home/xcpan/LRW/LRW.h5', "r")
# vidInp = cv.imdecode(h5["png"][0], cv.IMREAD_COLOR)
# vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
# plt.imshow(vidInp[0],cmap ='gray')
# plt.show()
#
# # dataset split
# dt = h5py.vlen_dtype(np.dtype('uint8'))
#
# f1 = h5py.File('/home/pychen/dataset/LRW/LRW_train.h5', 'a')
# dataset1 = f1.create_dataset('png', (488766,), dtype=dt)
# for i in trange(0,488766):
#   new_data = h5['png'][i]
#   dataset1[i] = new_data
# f1.close()
#
# f2 = h5py.File('/home/pychen/dataset/LRW/LRW_valid.h5', 'a')
# dataset2 = f2.create_dataset('png', (25000,), dtype=dt)
# for i in trange(488766,513766):
#   new_data = h5['png'][i]
#   dataset2[i-488766] = new_data
# f2.close()
#
# f3 = h5py.File('/home/pychen/dataset/LRW/LRW_test.h5', 'a')
# dataset3 = f3.create_dataset('png', (25000,), dtype=dt)
# for i in trange(513766,538766):
#   new_data = h5['png'][i]
#   dataset3[i-513766] = new_data
# f3.close()

# statistics
h5 = h5py.File('/home/xcpan/LRW/LRW.h5', "r")
gray = 0
for i in trange(len(h5['png'])):
  vidInp = cv.imdecode(h5["png"][i], cv.IMREAD_COLOR)
  vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
  vidInp = np.divide(vidInp,255)
  gray += np.sum(vidInp)
gray_mean = gray / (len(h5['png'])*29*120*120)

gray_var = 0
for i in trange(len(h5['png'])):
  vidInp = cv.imdecode(h5["png"][i], cv.IMREAD_COLOR)
  vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
  vidInp = np.divide(vidInp, 255)
  gray_var += np.sum((vidInp - gray_mean) ** 2)

gray_var = np.sqrt(gray_var/(len(h5['png'])*29*120*120))
print(gray_mean, gray_var)