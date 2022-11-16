import sys
import time

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow_datasets as tfds
# %matplotlib inline
tf.config.set_visible_devices([], 'GPU')

model = tf.keras.applications.mobilenet.MobileNet(input_shape=(32, 32, 3), include_top=False)

# st comments
import numpy as np

# Loading mnist dataset, that retrieves Image directly from datasset
# For processing the existing image from the data set, I put in this code. Can you review?

ds = tfds.load('cifar10', split='train')
ds = ds.take(5000)  # Only take a single example
# images_list = []



for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    print(list(example.keys()))
    image = example["image"]
    image = image.numpy().astype("float32")

    # img = tf.keras.preprocessing.image.array_to_img(img_data)
    # x = tf.keras.preprocessing.image.img_to_array(image)
    print("image array", image.shape)
    x = np.expand_dims(image, axis=0)
    y = preprocess_input(image)
    # images_list.append(y)

# reshapedImage = np.asarray(images_list)

testds = tfds.load('cifar10', split='test')
testds = testds.take(10)

for item in tfds.as_numpy(testds):
    image= item['image']
    # assert isinstance(image,np.array)
    print(image.shape)
    start = time.time()
    preds = model.predict(image.reshape(1,32,32,3))
    print('print predictions', preds)

    print("--- %s seconds ---" % (time.time() - start))
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=1))
