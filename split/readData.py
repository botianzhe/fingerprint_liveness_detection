import os
import cv2
import numpy as np
from tensorflow.python.framework import dtypes
from six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import math

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 2

def extract_train_images():
     train_set_x = []

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Live/'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Live/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          train_set_x.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/BodyDouble'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/BodyDouble/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          train_set_x.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Latex'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Latex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          train_set_x.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Playdoh'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Playdoh/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          train_set_x.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/WoodGlue'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/WoodGlue/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          train_set_x.append(res1_1_1)


     npdata = np.array(train_set_x)
     npdata = npdata.reshape(2221, IMAGE_SIZE, IMAGE_SIZE, 1)
     return npdata


def generate_train_labels():
     label = np.zeros(2221)
     offset = np.arange(1221)
     label.flat[offset] = 1
     label = label.astype(int)

     return label


def extract_eval_image():
     eval_set = []

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Live/'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Live/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)


     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/BodyDouble'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/BodyDouble/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Latex'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Latex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Playdoh'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/Playdoh/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/WoodGlue'):
          img1 = cv2.imread("/home/hkp/project/Data/LivDet2013/Training/Swipe/Spoof/WoodGlue/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)


     npdata = np.array(eval_set)
     npdata = npdata.reshape(2221, IMAGE_SIZE, IMAGE_SIZE, 1)
     return npdata

def generate_eval_label():
     label = np.zeros(2221)
     offset = np.arange(1221)
     label.flat[offset] = 1
     label = label.astype(int)

     return label

def read_data_set(validation_size=100, dtype=dtypes.float32,
                   reshape=True):
     train_image = extract_train_images()
     train_label = generate_train_labels()

     eval_image = extract_eval_image()
     eval_label = generate_eval_label()

     validation_images = train_image[:validation_size]
     validation_labels = train_label[:validation_size]

     train_image = train_image[validation_size:]
     train_label = train_label[validation_size:]

     train = DataSet(train_image, train_label, dtype=dtype, reshape=reshape)
     validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
     test = DataSet(eval_image, eval_label, dtype=dtype, reshape=reshape)

     return base.Datasets(train=train, validation=validation, test=test)