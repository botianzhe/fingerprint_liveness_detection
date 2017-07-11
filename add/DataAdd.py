import os
import cv2
import numpy as np
from tensorflow.python.framework import dtypes
from six.moves import xrange


IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 2

def lightchange(fn):
    img = cv2.imread(fn)
    w = img.shape[1]
    h = img.shape[0]
    ii = 0
    # change the light level
    for xi in xrange(0, w):
        for xj in xrange(0, h):
            # set the pixel value increase to 1020%
            img[xj, xi, 0] = int(img[xj, xi, 0] * 1.1)
            img[xj, xi, 1] = int(img[xj, xi, 1] * 1.1)
            img[xj, xi, 2] = int(img[xj, xi, 2] * 1.1)
            # show the process
    return cv2.bitwise_not(img)

def add(imgpath):
   	 for filenames in os.listdir(imgpath):
                    img = cv2.imread(imgpath+"/" + filenames, cv2.IMREAD_GRAYSCALE)
                    # change the size
                    res = cv2.resize( img , (104,750) ,interpolation=cv2.INTER_CUBIC)
                     #horizontal flip
                    flip_hor = cv2.flip(img, 1)
                     # change the light level
                    lightch=lightchange(imgpath+"/" + filenames)
                     # save
                    cv2.imwrite(imgpath+"/size_change_"+filenames,res)
                    cv2.imwrite(imgpath+"/flip_hor_"+filenames,flip_hor)
                    cv2.imwrite(imgpath+"/light_change_"+filenames,lightch)
