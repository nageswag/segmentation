import numpy as np
import matplotlib.pyplot as plt
import cv2

import pywt
import pywt.data


image = plt.imread('/home/workspace/assignment/dataset/cityscapes/trainValTest/leftImg8bit/val/frankfurt/frankfurt_000001_054219_leftImg8bit.png')
# Load image
Y = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
M, N = Y.shape
    
# Crop input image to be 3 divisible by 2
Y = Y[0:int(M/16)*16, 0:int(N/16)*16]

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(Y, 'haar')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
plt.savefig('/home/nga2abt/workspace/assignment/classical_segmentation/wavelet_out.png')
