import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import cv2
import os
train_dir = './dc_2000'
X = np.zeros((2000,75,75,3),dtype=np.float32)
y = np.zeros((2000,1),dtype=np.int8)
for i,j in enumerate(os.listdir(train_dir+'/train/cat')):
    img = cv2.imread(train_dir+'/train/cat/'+j)
    img = cv2.resize(img,(75,75))
    X[i] = img
    y[i] = 0#cat = 0
for i, j in enumerate(os.listdir(train_dir + '/train/dog')):
    img = cv2.imread(train_dir + '/train/dog/' + j)
    img = cv2.resize(img, (75, 75))
    X[1000+i] = img
    y[1000+i] = 1#dog = 1
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
