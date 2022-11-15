import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline

from tensorflow.keras.applications import imagenet_utils

# Organize data into train, valid, test dirs
os.chdir('data/Sign-Language-Digits')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
os.chdir('../..')

for i in range(0, 10):
    assert len(os.listdir(f'data/Sign-Language-Digits/valid/{i}')) == 30
    assert len(os.listdir(f'data/Sign-Language-Digits/test/{i}')) == 5

train_path = 'data/Sign-Language-Digits/train'
valid_path = 'data/Sign-Language-Digits/valid'
test_path = 'data/Sign-Language-Digits/test'


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

assert train_batches.n == 1712
assert valid_batches.n == 300
assert test_batches.n == 50
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 10


mobile = tf.keras.applications.mobilenet.MobileNet()

mobile.summary()

# x = mobile.layers[-7].output
# output = Dense(units=10, activation='softmax')(x)
#
# model = Model(inputs=mobile.input, outputs=output)
#
# for layer in model.layers[:-23]:
#     layer.trainable = False
#
# model.summary()

# mobile.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Run for more epochs (~30) to see better results
# mobile.fit(x=train_batches,  validation_data=valid_batches,epochs=10,verbose=2)

test_labels = test_batches.classes
predictions = mobile.predict(x=test_batches, steps=len(test_batches), verbose=0)

results = imagenet_utils.decode_predictions(predictions)
print(results)
# cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
# test_batches.class_indices

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')



