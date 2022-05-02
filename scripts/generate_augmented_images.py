# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os,sys
from PIL import Image


def generate_flipped_images(imgs_dir, gt_imgs_dir):
    
    names = []
    
    for name in os.listdir(imgs_dir):
        names.append(name)
        
    for img_name in names:
        
        img = Image.open(imgs_dir + img_name)
        gt_img = Image.open(gt_imgs_dir + img_name)
        
        img_id = img_name.rstrip('.png')
        
        vf_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        vf_img.save(imgs_dir + img_id + '_vf.png')
        
        hf_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        hf_img.save(imgs_dir + img_id + '_hf.png')
        
        vf_hf_img = vf_img.transpose(Image.FLIP_TOP_BOTTOM)
        vf_hf_img.save(imgs_dir + img_id + '_vf_hf.png')
        
        vf_gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
        vf_gt_img.save(gt_imgs_dir + img_id + '_vf.png')
        
        hf_gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)
        hf_gt_img.save(gt_imgs_dir + img_id + '_hf.png')
        
        vf_hf_gt_img = vf_gt_img.transpose(Image.FLIP_TOP_BOTTOM)
        vf_hf_gt_img.save(gt_imgs_dir + img_id + '_vf_hf.png')
        
        
def generate_transposed_images(imgs_dir, gt_imgs_dir):
    
    
    names = []
    
    for name in os.listdir(imgs_dir):
        names.append(name)
        
    for img_name in names:
        
        img = Image.open(imgs_dir + img_name)
        gt_img = Image.open(gt_imgs_dir + img_name)
        
        img_id = img_name.rstrip('.png')
        
        tp_img = img.transpose(Image.ROTATE_90)
        tp_img.save(imgs_dir + img_id + '_tp.png')
        
        tp_gt_img =  gt_img.transpose(Image.ROTATE_90)
        tp_gt_img.save(gt_imgs_dir + img_id + '_tp.png')
        
        
def generate_rotated_by_45_degrees_images(imgs_dir, gt_imgs_dir):
     
    n = 400
    names = []
    
    for name in os.listdir(imgs_dir):
        names.append(name)
        
    for img_name in names:
        
        img = Image.open(imgs_dir + img_name)
        gt_img = Image.open(gt_imgs_dir + img_name)
        
        img_canvas = Image.new('RGB', (3*n, 3*n))
        gt_img_canvas = Image.new('RGB', (3*n, 3*n))
        
        img_id = img_name.rstrip('.png')
        
        img_canvas.paste(img, (400,400)) #middle original
        img_canvas.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (800,400)) #right vertical flip
        img_canvas.paste(img.transpose(Image.FLIP_LEFT_RIGHT), (0,400)) #left vertical flip
        img_canvas.paste(img.transpose(Image.FLIP_TOP_BOTTOM), (400,800)) #down horizontal flip
        img_canvas.paste(img.transpose(Image.FLIP_TOP_BOTTOM), (400,0)) #up horizontal flip
    
        rotated_img = img_canvas.rotate(45).crop((400, 400, 800, 800))        
        rotated_img.save(imgs_dir + img_id + '_rot.png')
        
        gt_img_canvas.paste(gt_img, (400,400)) #middle original
        gt_img_canvas.paste(gt_img.transpose(Image.FLIP_LEFT_RIGHT), (800,400)) #right vertical flip
        gt_img_canvas.paste(gt_img.transpose(Image.FLIP_LEFT_RIGHT), (0,400)) #left vertical flip
        gt_img_canvas.paste(gt_img.transpose(Image.FLIP_TOP_BOTTOM), (400,800)) #down horizontal flip
        gt_img_canvas.paste(gt_img.transpose(Image.FLIP_TOP_BOTTOM), (400,0)) #up horizontal flip
    
        rotated_gt_img = gt_img_canvas.rotate(45).crop((400, 400, 800, 800))        
        rotated_gt_img.save(gt_imgs_dir + img_id + '_rot.png')
        
        
  
    
        
        
        
        
        
        
        
        
        
        
        