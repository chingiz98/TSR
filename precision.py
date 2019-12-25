import os
import keras as keras
import numpy as np
from math import sqrt
import imutils
import skimage
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from skimage import io
from skimage import transform
from PIL import Image


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob
import cv2
import os
import keras as keras
from sklearn.model_selection import train_test_split
from skimage.color import rgb2grey


model = tf.keras.models.load_model('new_model1.h5')
labelNames = open("signnames1.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)]
        for f in file_names:
            images2 = []
            image = io.imread(f)
            #images.append(io.imread(f))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("test", image)
            image = rgb2grey(image)
            #cv2.imshow("test1", image)
            image = (image / 255.0)  # rescale
            image = cv2.resize(image, (32, 32))  # resize
            images2.append(image)
            images2 = np.stack([img[:, :, np.newaxis] for img in images2], axis=0).astype(np.float32)

            sas = np.reshape(images2[0], (1, 32, 32, 1))

            preds = model.predict(sas)
            j = preds.argmax(axis=1)[0]
            label = labelNames[j]
            print(label)




load_data("rr")





