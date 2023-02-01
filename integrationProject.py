import sys
import time

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# %matplotlib inline
tf.config.set_visible_devices([], 'GPU')
model = MobileNetV3Large(weights='imagenet')
# st comments
import numpy as np

# In this code , we are loading image from the disc and preprocessing this using keras utility
if len(sys.argv) < 2:
    print("Please specify image")
# img_path = sys.argv[1]
img_path = 'flower.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Loading mnist dataset, that retrieves Image directly from datasset
# For processing the existing image from the data set, I put in this code. Can you review?

# ds = tfds.load('mnist', split='train')
# ds = ds.take(10)  # Only take a single example
#
# for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
#     print(list(example.keys()))
#     image = example["image"]
#     image = image.numpy().astype("float32")
#
#     # img = tf.keras.preprocessing.image.array_to_img(img_data)
#     # x = tf.keras.preprocessing.image.img_to_array(image)
#     print("image array", image)
#     x = np.expand_dims(image, axis=0)
#     x = preprocess_input(image)

for i in range(10):
    start = time.time()
    preds = model.predict(x)
    print("--- %s seconds ---" % (time.time() - start))
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
