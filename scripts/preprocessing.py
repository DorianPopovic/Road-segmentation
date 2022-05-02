import numpy as np
from helpers import *

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename,patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X

#classifying ground truth patches:
def value_to_class(v):
    foreground_threshold=0.25
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

# Extract class from ground truth image
def ground_truth(filename,patch_size):
    gt_img = load_image(filename)
    img_patches = img_crop(gt_img, patch_size, patch_size)
    X = np.asarray([value_to_class(np.mean(img_patches[i])) for i in range(len(img_patches))])
    return X