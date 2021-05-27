import numpy as np
import cv2
from numpy.lib.arraysetops import intersect1d
from keras import backend as K
from scipy import stats

# IoU 
def jaccard_index(predicted_image, gt_image):
    intersection = np.logical_and(gt_image, predicted_image)
    union = np.logical_or(gt_image, predicted_image)
    smooth = 0.01
    iou_score = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou_score

# dice
def dice_coef(predicted_image, gt_image):
    jaccard = jaccard_index(predicted_image, gt_image)
    dice_score = 2 * jaccard / (1 + jaccard)
    # alternative
    #intersection = np.logical_and(gt_image, predicted_image)
    #smooth = 0.01
    #dice_score = 2 * (np.sum(intersection) + smooth)  / (np.sum(gt_image) + np.sum(predicted_image) + smooth)
    return dice_score

# pixel accuracy
def pixel_metric(predicted_image, gt_image):
    numerator = np.sum(np.logical_and(gt_image, predicted_image))
    denominator = np.sum(gt_image)
    pixel_score = numerator / denominator
    return pixel_score

# rank correlation
def spearman_correlation(metric_1_rank, metric_2_rank):
    jd_correlation = stats.spearmanr(metric_1_rank, metric_2_rank)
    print("jaccard_dice_correlation", jd_correlation)


