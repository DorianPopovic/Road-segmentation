

import matplotlib.pyplot as plt


from sklearn import linear_model
import sklearn as sk
import sklearn.neighbors as neighb
from helpers import *
from preprocessing import *
from mask_to_submission import *
from imaging import *
import numpy as np
import matplotlib.image as mpimg
import zipfile

with zipfile.ZipFile('../data/training.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')
with zipfile.ZipFile('../data/test_set_images.zip', 'r') as zip_ref:
    zip_ref.extractall('../data')

n=100 #number of images used for validation and training

imgs, gt_imgs = load_images(n)
print(np.array(imgs).shape)

# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

print(np.array(img_patches).shape)

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

print(img_patches.shape)

#Extracting features
X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(len(set(Y)) ))


Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')

'''
#Testing for different k numbers, and fitting each:
for i in range(10):
    model= neighb.KNeighborsClassifier(n_neighbors=1+2*i)
    model.fit(X,Y)
    print('score:',model.score(X,Y))
'''

#Fitting linear regression
model=linear_model.LinearRegression()
model.fit(X,Y)


make_submission(model)
