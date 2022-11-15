
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing. image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from  sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import MobileNetV3Large
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

# %matplotlib inline
tf.config.set_visible_devices([], 'GPU')
# mobile = tf.keras.applications.mobilenet.MobileNet()
mobile = MobileNetV3Large(weights='imagenet')
def prepare_image(file):
    image_path='data/MobileNet-samples/'
    img = image.load_img(image_path+ file, target_size=(224,224))
    img_array=image.img_to_array(img)
    image_array_exapand_dims = np.expand_dims(img_array,axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(image_array_exapand_dims)

from IPython.display import Image
Image(filename='data/MobileNet-samples/elephant.jpg', width=300,height=200)

preprocessed_image= prepare_image('flower.jpg')
predictions  = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

print(results)