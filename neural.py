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

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(io.imread(f))
            labels.append(int(d))
    images = np.array(images)
    labels = np.array(labels)
    images28 = [transform.resize(image, (28, 28)) for image in images]
    images28 = np.array(images28)
    images28 = skimage.color.rgb2gray(images28)
    images28 = np.array(images28)
    return images28, labels

train_data_directory = "Training"
test_data_directory = "Testing"

# load trainning data

# images, labels = load_data(train_data_directory)
# train_x = np.reshape(images, (4575, 784))
# train_y = keras.utils.to_categorical(labels, 62)
#
# # # load test data
# images, labels = load_data(test_data_directory)
# test_x = np.reshape(images, (2520, 784))
# test_y = keras.utils.to_categorical(labels, 62)
#
# model = Sequential()
# model.add(Dense(units = 128, activation="relu", input_shape = (784,)))
# model.add(Dense(units = 128, activation="relu"))
# model.add(Dense(units = 128, activation="relu"))
# model.add(Dense(units=62,activation="softmax"))
# model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
# model.fit(train_x,train_y, batch_size=32,epochs=200,verbose=2)
# score = model.evaluate(test_x, test_y, batch_size=32)
# print(score)
# model.save('model2.h5')

model = keras.models.load_model('model.h5')

input_img = io.imread("test_img.ppm")
output_image = skimage.transform.resize(input_img, (784, 784))
img = skimage.color.rgb2gray(np.array(output_image))

sas = model.predict(np.array(img))

print(sas)
print(np.argmax(sas[0]))