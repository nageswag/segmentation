import matplotlib.pyplot as plt
from scipy.spatial.distance import dice
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import cv2
import png
from sklearn.cluster import KMeans

gt_pic = plt.imread('/home/nga2abt/workspace/assignment/dataset/cityscapes/gt/gtFine/val/frankfurt/frankfurt_000001_054219_gtFine_labelIds.png')
#gt_pic = plt.imread('/home/nga2abt/workspace/assignment/dataset/cityscapes/gt/gtFine/train/stuttgart/stuttgart_000059_000019_gtFine_color.png')
print(gt_pic.shape)
pic = plt.imread('/home/nga2abt/workspace/assignment/dataset/cityscapes/trainValTest/leftImg8bit/val/frankfurt/frankfurt_000001_054219_leftImg8bit.png')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)

# reshape test image
test_image = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
print("test_image dimension check", test_image.shape)

gt_test_image = gt_pic.reshape(gt_pic.shape[0], gt_pic.shape[1])
print("gt_test_image dimension check", gt_test_image.shape)

# perform k-means clustering
kmeans = KMeans(n_clusters=4, random_state=10).fit(test_image)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
print(pic2show.shape)

#cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
cluster_pic = kmeans.labels_.reshape(pic.shape[0], pic.shape[1])

# plot
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.imshow(cluster_pic)

plt.show()
plt.savefig('out.png')