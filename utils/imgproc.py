import numpy as np
import cv2

def normalization(img,min_val = 0,max_val = 1):
    norm_img = (img - np.min(img)) * ((max_val - min_val) / (np.max(img) - np.min(img))) + min_val
    return norm_img

def standardize(img):
    return (img - img.mean()) / img.std()

def normalizationPerChannel(img):
    ##### channel dimension is assumed to be the last dimension of the image array
    ch = img.shape[-1]
    for i in range(ch):
        img[...,i] = normalization(img[...,i])
    return img

def standardizePerChannel(img):
    ##### channel dimension is assumed to be the last dimension of the image array
    ch = img.shape[-1]
    for i in range(ch):
        img[...,i] = standardize(img[...,i])
    return img

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def bgr2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


