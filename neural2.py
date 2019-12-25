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

NUM_CLASSES = 43
np.random.seed(42)


# path to the images
data_path = 'GTSRB/Final_Training/Images'

images = []
image_labels = []

# get the image paths
# for i in range(NUM_CLASSES):
#     image_path = data_path + '/' + format(i, '05d') + '/'
#     for img in glob.glob(image_path + '*.ppm'):
#         image = cv2.imread(img)
#         image = rgb2grey(image)
#         image = (image / 255.0)  # rescale
#         image = cv2.resize(image, (32, 32))  # resize
#         images.append(image)
#
#         # create the image labels and one-hot encode them
#         labels = np.zeros((NUM_CLASSES,), dtype=np.float32)
#         labels[i] = 1.0
#         image_labels.append(labels)





model = tf.keras.models.load_model('new_model1.h5')
# sas = np.reshape(images[120], (1,32,32,1))
# cv2.imshow("test", images[120])
# ses = model.predict(sas)
# print(np.argmax(ses[0]))


image = cv2.imread("test3.jpg")
image = rgb2grey(image)
cv2.imshow("test1", image)
image = (image / 255.0)  # rescale
image = cv2.resize(image, (32, 32))  # resize
images.append(image)

images = np.stack([img[:, :, np.newaxis] for img in images], axis=0).astype(np.float32)
#image_labels = np.matrix(image_labels).astype(np.float32)
sas = np.reshape(images[0], (1,32,32,1))
#cv2.imshow("test", images[120])
ses = model.predict(sas)
print(np.argmax(ses[0]))


cv2.waitKey()
cv2.destroyAllWindows()

# plt.imshow(images[45, :, :, :].reshape(32, 32), cmap='gray')
# print(image_labels[45, :])
#
#
# # divide the data into train and test set
# (train_X, test_X, train_y, test_y) = train_test_split(images, image_labels,
#                                                       test_size=0.2,
#                                                       random_state=42)
# print(train_X.shape)
# print(train_y.shape)
# print(test_X.shape)
# print(test_y.shape)
#
# # initialize the model
# model = tf.keras.models.Sequential()
# input_shape = (32, 32, 1)  # grey-scale images of 32x32
#
# model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same',
#                                  activation='relu', input_shape=input_shape))
# model.add(tf.keras.layers.BatchNormalization(axis=-1))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same',
#                                  activation='relu'))
# model.add(tf.keras.layers.BatchNormalization(axis=-1))
# model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same',
#                                  activation='relu'))
# model.add(tf.keras.layers.BatchNormalization(axis=-1))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.4))
#
# model.add(tf.keras.layers.Dense(43, activation='softmax'))
#
#
#
#
#
# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer,
#               metrics=['accuracy'])
#
# history = model.fit(train_X, train_y,
#                     validation_data=(test_X, test_y),
#                     epochs=4)
#
# model.save("new_model.h5")
#
# num_epochs = np.arange(0, 10)
# plt.figure(dpi=300)
# plt.plot(num_epochs, history.history['loss'], label='train_loss', c='red')
# plt.plot(num_epochs, history.history['val_loss'],
#     label='val_loss', c='orange')
# plt.plot(num_epochs, history.history['acc'], label='train_acc', c='green')
# plt.plot(num_epochs, history.history['val_acc'],
#     label='val_acc', c='blue')
# plt.title('Training Loss and Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Loss/Accuracy')
# plt.legend()
# plt.savefig('plot.png')

