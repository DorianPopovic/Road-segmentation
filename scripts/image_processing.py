import cv2 as cv
import numpy as np
import os
from mask_to_submission import *
from tqdm import tqdm
from PIL import Image

def montage(im1, im2, im3):
    '''Creates a single image with the aerial image, the prediction, and the floodfilled prediction
    Keyword arguments:
    im1, im2, im3: The 3 images to merge together
    '''
    images = [Image.open(x) for x in [im1, im2, im3]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0

    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def open_image(im_in):
    '''Operator for opening the image and filling holes in the road prediction 
    Keyword arguments:
    im_in: The imput image, to be opened
    '''
    im_th = cv.bitwise_not(im_in)
    #th, im_th = cv.threshold(im_in, 125, 255, cv.THRESH_BINARY_INV)
    DISC = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    im_disc = cv.morphologyEx(im_th, cv.MORPH_OPEN, DISC)
    im_out = cv.bitwise_not(im_disc)

    return im_out

def floodfill_image(im_in):
    '''Operator for floodfilling the image, gets rid of isolated white spots on the prediction image.
    Keyword arguments:
    im_in: The imput image, to be floodfilled
    '''
    im_inv = cv.bitwise_not(im_in)

    th, im_th = cv.threshold(im_inv, 250, 255, cv.THRESH_BINARY)

    im_floodfill = im_th.copy()

    h,w = im_inv.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)


    for row, r in zip(im_inv.T[:], range(h)):
        if row[0] != 255:
            cv.floodFill(im_floodfill, mask, (r, 0), 255)
        if row[h-1] != 255:
            cv.floodFill(im_floodfill, mask, (r, w-1), 255)       
            
            
    for col, c in zip(im_inv[:], range(w)):
        if col[0] != 255:
            cv.floodFill(im_floodfill, mask, (0, c), 255)
        if col[w-1] != 255:
            cv.floodFill(im_floodfill, mask, (w-1, c), 255)
                            
    im_floodfill_inv = cv.bitwise_not(im_floodfill)    

    final = im_floodfill_inv | cv.bitwise_not(im_in)

    im_out = cv.bitwise_not(final)

    return im_out


mod_dir = '../data/sub_mod/'
if not os.path.isdir(mod_dir):
    os.mkdir(mod_dir)

predictions = []
submission_filename = '../data/mod_sub.csv'

print('Processing images')

for i in tqdm(range(1,51)):
    image_filename = '../data/submissionimages/prediction' + '%.3d' % i + '.png'

    im_in = cv.imread(image_filename, cv.IMREAD_GRAYSCALE)

    im_open = open_image(im_in)
    im_flood = floodfill_image(im_open)

    mod_path = mod_dir + 'modified_' + '%.3d' % i + '.png'
    cv.imwrite(mod_path, im_flood)

    predictions.append(mod_path)

masks_to_submission(submission_filename, *predictions)

print('Building montage')

mont_dir = '../data/mont_dir/'
if not os.path.isdir(mont_dir):
    os.mkdir(mont_dir)
    
for i in tqdm(range(1,51)):
    im_1 = "../data/test_set_images/test_" + str(i) + "/test_" + str(i) + ".png"
    im_2 = '../data/submissionimages/prediction' + '%.3d' % i + '.png'
    im_3 = mod_dir + 'modified_' + '%.3d' % i + '.png'

    
    new_im = montage(im_1, im_2, im_3)

    new_im.save(mont_dir + 'modified_' + '%.3d' % i + '.png')
