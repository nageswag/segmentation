from __future__ import print_function

import argparse
import cv2
import numpy as np
from metrics_supervised import jaccard_index, dice_coef, pixel_metric, spearman_correlation
import tensorflow as tf
from dataset_parser.generator import get_gt_result_map

from model.unet import unet
from model.fcn import fcn_8s
from model.pspnet import pspnet50

# mask gt image including all classes for comparing with result
def mask_gt_image_color(res_map):  
    img = np.zeros((256, 512, 3), dtype=np.uint8)

    # For np.where calculation.
    res_map = np.squeeze(res_map)
    # take only the red channel value as RGB are same
    res_map_one_channel = res_map[:, :, 2]
 
    person = (res_map_one_channel == 24)
    car = (res_map_one_channel == 26)
    road = (res_map_one_channel == 7)

    img[:, : , 0] = np.where(person, 255, 0)
    img[:, :, 1] = np.where(car, 255, 0)
    img[:, :, 2] = np.where(road, 255, 0)

    return img

# class wise masking of gt image
def mask_gt_image(res_map):
    img = np.zeros((256, 512, 3, 3), dtype=np.uint8)
    res_gt = res_map[:, :, 2]
    #res_map = np.squeeze(res_gt)

    # For np.where calculation.
    person = (res_gt == 24)
    car = (res_gt == 26)
    road = (res_gt == 7)
    
    img[:, :, 0, 0] = np.where(person, 24, 0)
    img[:, :, 0, 1] = np.where(car, 26, 0)
    img[:, :, 0, 2] = np.where(road, 7, 0)

    img[:, :, 1, 0] = np.where(person, 24, 0)
    img[:, :, 1, 1] = np.where(car, 26, 0)
    img[:, :, 1, 2] = np.where(road, 7, 0)

    img[:, :, 2, 0] = np.where(person, 24, 0)
    img[:, :, 2, 1] = np.where(car, 26, 0)
    img[:, :, 2, 2] = np.where(road, 7, 0)

    return img

def result_map_to_img(res_map):
    img = np.zeros((256, 512, 3, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)

    argmax_idx = np.argmax(res_map, axis=2)

    # For np.where calculation.
    person = (argmax_idx == 1)
    car = (argmax_idx == 2)
    road = (argmax_idx == 3)

    img[:, :, 0, 0] = np.where(person, 24, 0)
    img[:, :, 0, 1] = np.where(car, 26, 0)
    img[:, :, 0, 2] = np.where(road, 7, 0)

    img[:, :, 1, 0] = np.where(person, 24, 0)
    img[:, :, 1, 1] = np.where(car, 26, 0)
    img[:, :, 1, 2] = np.where(road, 7, 0)

    img[:, :, 2, 0] = np.where(person, 24, 0)
    img[:, :, 2, 1] = np.where(car, 26, 0)
    img[:, :, 2, 2] = np.where(road, 7, 0)

    return img

def result_map_to_img_color(res_map):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)

    argmax_idx = np.argmax(res_map, axis=2)

    # For np.where calculation.
    person = (argmax_idx == 1)
    car = (argmax_idx == 2)
    road = (argmax_idx == 3)

    img[:, :, 0] = np.where(person, 255, 0)
    img[:, :, 1] = np.where(car, 255, 0)
    img[:, :, 2] = np.where(road, 255, 0)

    return img


# Parse Options
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", required=True, choices=['fcn', 'unet', 'pspnet'],
                    help="Model to test. 'fcn', 'unet', 'pspnet' is available.")
parser.add_argument("-P", "--img_path", required=True, help="The image path you want to test")

args = parser.parse_args()
model_name = args.model
img_path = args.img_path

# Using only 3 classes.
labels = ['background', 'person', 'car', 'road']

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet":
    model = unet(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)

try:
    model.load_weights(model_name + '_model_weight.h5')
except:
    print("You must train model and get weight before test.")

# reshape test image to match network input
x_img = cv2.imread(img_path)
r = 256.0 / x_img.shape[0]
dim = (int(x_img.shape[1] * r), 256)
x_img = cv2.resize(x_img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('x_img', x_img)
cv2.imwrite('frankfurt_000001_054219_test_image.png', x_img)

# reshape gt image for evaluation
gt = cv2.imread('/home/workspace/assignment/dataset/cityscapes/gt/gtFine/val/frankfurt/frankfurt_000001_054219_gtFine_labelIds.png')
r = 256.0 / x_img.shape[0]
dim = (int(x_img.shape[1] * r), 256)
gt_image = cv2.resize(gt, dim, interpolation = cv2.INTER_NEAREST)
cv2.imshow('gt_img', gt_image)
cv2.imwrite('berlin_000100_000019_gt_image.png', gt_image)

# mask for class individual extraction
gt_compare = mask_gt_image(gt_image)
cv2.imshow('gt_result_car', gt_compare[:, :, :, 1])
print("gt_compare_shape", gt_compare.shape)
cv2.imwrite('frankfurt_000001_054219_gt_compare_car.png', gt_compare[:, :, :, 1])
cv2.imwrite('frankfurt_000001_054219_gt_compare_road.png', gt_compare[:, :, :, 2])

# mask for whole image for all classes
gt_color = mask_gt_image_color(gt_image)
cv2.imshow('gt_color', gt_color)
cv2.imwrite('frankfurt_000001_054219_gt_color.png', gt_color)

x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

pred = model.predict(x_img)
cv2.imshow('pred', pred[0])
res_color = result_map_to_img_color(pred[0])
res_compare = result_map_to_img(pred[0])
cv2.imshow('res_color', res_color)
cv2.imshow('res_compare_car', res_compare[:, :, :, 1])
cv2.imwrite('frankfurt_000001_054219_unet_prediction.png', pred[0])
cv2.imwrite('frankfurt_000001_054219_unet_res_color.png', res_color)
cv2.imwrite('frankfurt_000001_054219_unet_result_car.png', res_compare[:, :, :, 1])
cv2.imwrite('frankfurt_000001_054219_unet_result_road.png', res_compare[:, :, :, 2])

# compute metrics
# jaccard overall
jaccard_overall_score = jaccard_index(res_color, gt_color)
print("jaccard_overall_score", jaccard_overall_score)
# jaccard for car class
jaccard_car_score = jaccard_index(res_compare[:, :, 2, 1], gt_compare[:, :, 2, 1])
print("jaccard_car_score", jaccard_car_score)
#jaccard for person
jaccard_person_score = jaccard_index(res_compare[:, :, 2, 0], gt_compare[:, :, 2, 0])
print("jaccard_person_score", jaccard_person_score)
#jaccard for road
jaccard_road_score = jaccard_index(res_compare[:, :, 2, 2], gt_compare[:, :, 2, 2])
print("jaccard_road_score", jaccard_road_score)

# dice coeff
dice_overall_score = dice_coef(res_color, gt_color)
print("dice_overall_score", dice_overall_score)
# dice for car class
dice_car_score = dice_coef(res_compare[:, :, 2, 1], gt_compare[:, :, 2, 1])
print("dice_car_score", dice_car_score)
# dice for person
dice_person_score = dice_coef(res_compare[:, :, 2, 0], gt_compare[:, :, 2, 0])
print("dice_person_score", dice_person_score)
# dice for road
dice_road_score = dice_coef(res_compare[:, :, 2, 2], gt_compare[:, :, 2, 2])
print("dice_road_score", dice_road_score)


# pixel accuracy
pixel_overall_score = pixel_metric(np.where(res_color==255, 1, 0), np.where(gt_color==255, 1, 0))
print("pixel_overall_score", pixel_overall_score)
#pixel accuracy car
pixel_car_score = pixel_metric(np.where(res_compare[:, :, 2, 1], 1, 0), np.where(gt_compare[:, :, 2, 1], 1, 0))
print("pixel_car_score", pixel_car_score)
#pixel accuracy car
pixel_person_score = pixel_metric(np.where(res_compare[:, :, 2, 0], 1, 0), np.where(gt_compare[:, :, 2, 0], 1, 0))
print("pixel_person_score", pixel_person_score)
#pixel accuracy car
pixel_road_score = pixel_metric(np.where(res_compare[:, :, 2, 2], 1, 0), np.where(gt_compare[:, :, 2, 2], 1, 0))
#print("test", np.unique(np.where(res_compare[:, :, 2, 2], 1, 0)))
print("pixel_road_score", pixel_road_score)

# get statistical dependence of rank between jaccard and dice for u-net as an example
jaccard_score = [0.8380576120625918,  0.7940876764114234, 3.27858103013016e-05, 0.8446160697058234]
dice_score = [0.9118948248005767, 0.8852272794156597, 6.556947085437022e-05, 0.9157635386322114]

jaccard_rank = [2, 3, 4, 1]
dice_rank = [2, 3, 4, 1]

jd_correlation = spearman_correlation(jaccard_score, dice_score)

cv2.waitKey(0)

