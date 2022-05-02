import numpy as np
import os
from preprocessing import *
from PIL import Image
from mask_to_submission import *

patch_size=16

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im

def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def make_submission(model):
    #creating submission images
    prediction_training_dir = "../data/submissionimages/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    for i in range(1, 51):
        image_filename = '../data/test_set_images/test_' + str(i) + '/test_'+ str(i) +'.png'

        test=extract_img_features(image_filename,patch_size)
        pred=model.predict(test)
        image = load_image(image_filename)
        w = image.shape[0]
        h = image.shape[1]
        predicted_im = label_to_img(w, h, patch_size, patch_size, pred)
        predicted_im = binary_to_uint8(predicted_im)
        Image.fromarray(predicted_im).save('../data/submissionimages/prediction' + '%.3d' % i + '.png')

    #creating submission file
    submission_filename = '../data/submission.csv'
    image_filenames = []

    for i in range(1, 51):
        image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'
        image_filenames.append(image_filename)

    masks_to_submission(submission_filename, *image_filenames)