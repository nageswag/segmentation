from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import cv2
import png
import matplotlib.pyplot as plt
from scipy import ndimage

gt_image = plt.imread('/home/workspace/assignment/dataset/cityscapes/gt/gtFine/val/frankfurt/frankfurt_000001_054219_gtFine_labelIds.png')
print("gt image", gt_image.shape)
image = plt.imread('/home/workspace/assignment/dataset/cityscapes/trainValTest/leftImg8bit/val/frankfurt/frankfurt_000001_054219_leftImg8bit.png')
plt.imshow(image)

# converting to grayscale
gray = rgb2gray(image)

# laplace operator
kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
l_result = ndimage.convolve(gray, kernel_laplace, mode='reflect')

# Laplacian of Gaussian (LoG)
LoG_result = ndimage.gaussian_laplace(gray, sigma=3)
print("LoG image dimensions check", LoG_result.shape)

# plot utilities
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(l_result, cmap='gray')
ax2 = fig.add_subplot(122)
ax2.imshow(LoG_result, cmap='gray')

plt.show()
plt.savefig('/home/workspace/assignment/classical_segmentation/laplace_out.png')
