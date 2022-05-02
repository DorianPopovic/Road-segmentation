from tensorflow.python.keras import models
import matplotlib.image as mpimg
import os
from PIL import Image
import sys
from mask_to_submission import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from U_net import *
from losses import *
import cv2
import zipfile
import helpers


with zipfile.ZipFile('../data/test_set_images.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')

PIXEL_DEPTH=255

def prediction_to_class(image):
    """returns binary output from greyscale image"""
    threshold = 0.5
    out=np.empty(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > threshold:
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out

def getprediction_mean(img,model):
    '''
    Creates 608x608 prediction by splitting the prediction into 9 zones and averaging when possible.
    '''
    #more logical but works less well than predict_test_img?

    #takes as input a 608x608 image and gives a 608x608 prediction based on 400x400 predictions from network
    subimgs = np.empty((4,400,400,3))

    #we create 4 different predictions

    subimgs[0]=img[0:400,0:400,:]
    subimgs[1]=img[0:400,208:608,:]
    subimgs[2]=img[208:608,0:400,:]
    subimgs[3]=img[208:608,208:608,:]
    preds=model.predict(subimgs)

    output=np.empty((608,608))
    #we first fill the 4 corners with the only predictions for the corners

    output[0:208,0:208]=preds[0,0:208,0:208,0]
    output[0:208,400:608] = preds[1, 0:208,192:400, 0]
    output[400:608,0:208] = preds[2, 192:400,0:208, 0]
    output[400:608, 400:608] = preds[3,192:400, 192:400, 0]

    #then we fill the 4 middle rectangles with means of 2 predictions
    recttop=np.empty((2,208,192))
    rectbottom = np.empty((2,  208,192,))
    rectleft = np.empty((2, 192,208))
    rectright = np.empty((2, 192,208))

    recttop[0]=preds[0,0:208,208:400,0]
    recttop[1] = preds[1,0:208,0:192, 0]
    output[0:208,208:400]=np.mean(recttop,axis=0)

    rectbottom[0] = preds[2,  192:400,208:400, 0]
    rectbottom[1] = preds[3,  192:400,0:192, 0]
    output[ 400:608,208:400] =np.mean(rectbottom, axis=0)


    rectleft[0] = preds[0, 208:400,0:208,  0]
    rectleft[1] = preds[2,  0:192,0:208, 0]
    output[ 208:400,0:208] = np.mean(rectleft, axis=0)

    rectright[0] = preds[1,  208:400,192:400, 0]
    rectright[1] = preds[3,  0:192,192:400, 0]
    output[ 208:400,400:608] = np.mean(rectright, axis=0)

    #then we fill the center with means of all predictions
    tomean=np.empty((4,192,192))
    tomean[0]=preds[0,208:400,208:400,0]
    tomean[1]=preds[1,208:400,0:192,0]
    tomean[2] = preds[2,  0:192,208:400, 0]
    tomean[3] = preds[3,  0:192,0:192, 0]
    output[208:400,208:400]=np.mean(tomean,axis=0)

    return output

def getprediction_resize(img, model):
    """
    Creates 608x608 prediction by resizing the input image and then resizing the output image
    """
    img=cv2.resize(img,(400,400))
    img=img.reshape((1,400,400,3))
    out=model.predict(img)
    out = out.reshape(( 400, 400, 1))
    out=cv2.resize(out,(608,608))
    return out

def getprediction_nomean(img, model):
    """
    Creates 608x608 prediction by splitting the prediction into 4 zones and taking the corresponding
    predictions
    """
    sub1 = img[:400, :400]
    sub2 = img[:400, 208:]
    sub3 = img[208:, :400]
    sub4 = img[208:, 208:]

    pred = model.predict(np.array(([sub1, sub2, sub3, sub4])))

    out = np.zeros((608, 608))
    out[:400, :400] = pred[0,:,:,0]
    out[:400, 208:] = pred[1,:,:,0]
    out[208:, :400] = pred[2, :, :, 0]
    out[208:, 208:] = pred[3,:,:,0]

    return out

save_model_path = '../results/weights_1200imgs_kernel5x5_100epochs.h5'
model = ResUNet(400,kernel_size=(5,5))
model.load_weights(save_model_path)

prediction_training_dir = "../data/submissionimages/"
if not os.path.isdir(prediction_training_dir):
    os.mkdir(prediction_training_dir)


for i in range(1, 51):
    test_data_filename=("../data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png")

    img=cv2.imread(test_data_filename)
    img=img/255.0 # we do this because the resnet is trained between 0 and 1
    pimg = getprediction_mean(img, model)
    pimg = img_float_to_uint8(pimg)
    Image.fromarray(pimg).save(prediction_training_dir + 'prediction' + '%.3d' % i + '.png')


# creating submission file
submission_filename = '../data/submission.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)