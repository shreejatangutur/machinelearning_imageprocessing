import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet,decode_predictions

import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras import optimizers
from keras.layers import Resizing
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D,BatchNormalization,LayerNormalization



# Loading the Dataset and getting size of it
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print("Size of the train images", x_train.shape)
print("Size of the test images", x_test.shape)


x_test, y_test = x_test[:5000], y_test[:5000]
x_val, y_val = x_train[:5000], y_train[:5000]
x_train, y_train = x_train[30000:], y_train[30000:]


print("Training data size: ", x_train.shape)
print("Validation data size: ", x_val.shape)
print("Test data size: ", x_test.shape)
print("Training data Labels", y_train.shape)
print("Validation data Labels", y_val.shape)



from keras.utils import np_utils

num_classes = 10

# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)


from sklearn.utils import shuffle

x_val, y_val  = shuffle(x_val, y_val)
x_train, y_train = shuffle(x_train, y_train)


# Base model for the mobilenet model
mobnet_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.





image_size = (224,224)

from keras import layers
from keras import models

num_classes = 10

model1 = Sequential()
keras.layers.experimental.preprocessing.Resizing(image_size[0], image_size[1], interpolation="bilinear", input_shape=x_train.shape[1:]),

model1.add(mobnet_model)
model1.add(GlobalAveragePooling2D())

model1.add(Dense(1024,activation=('relu')))
model1.add(Dense(512,activation=('relu')))
model1.add(Dense(256,activation=('relu')))
model1.add(Dropout(0.5))
model1.add(Dense(128,activation=('relu')))
model1.add(Dropout(0.5))
model1.add(Dense(10,activation=('softmax')))


model1.compile(loss='categorical_crossentropy',
              optimizer= 'Adam',
              metrics=['acc'])
# Creating the model and compiling it
model1.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# a summary of the model
model1.summary()


history = model1.fit(x_train, y_train, batch_size = 64, epochs = 2, validation_data = (x_test, y_test))



loss,accuracy = model1.evaluate(x_test,y_test)
print("Accuracy for test data : ",accuracy)
print("Loss for test data : ",loss)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import multilabel_confusion_matrix

rounded_labels=np.argmax(y_test, axis=1)

preds = model1.predict(x_test, batch_size = 64, verbose = 1)
preds = np.argmax(preds, axis=1) # to get the indices of max value in each row
cr = confusion_matrix(rounded_labels, preds)
print("The confusion Matrix: \n",cr)

# print('Predicted:', decode_predictions(preds, top=1))