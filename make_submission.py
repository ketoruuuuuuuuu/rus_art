import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
import pandas as pd
from PIL import Image


MODEL_PATH = "./cp3-002.keras"
TEST_DATASET = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"


IMG_SIZE = 224
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./127.5, offset = -1)
    ])

full_size = (224,224,3)

def read_image(img_file):
    img_file = str(img_file.numpy(),encoding='utf-8')
    img = Image.open(os.path.join(TEST_DATASET,img_file)).convert('RGB')
    img = np.array(img)
    img = tf.constant(img,dtype=tf.float32)
    img = resize_and_rescale(img, training=True)
    return img

def load(x):
    img = tf.py_function(read_image, [x], [tf.float32])
    img = img[0]
    img = tf.ensure_shape(img,full_size)
    return(img)

AUTOTUNE = tf.data.AUTOTUNE

def create_test_dataset(file_paths):
    #todo update filepaths to array from os.listdir()
    # file_paths = df['image_name'].values
    data_set = tf.data.Dataset.from_tensor_slices(file_paths)
    data_set = data_set.map(lambda x: load(x), num_parallel_calls=AUTOTUNE)
    data_set = data_set.batch(1)
    data_set = data_set.prefetch(buffer_size=AUTOTUNE)
    return data_set

if __name__ == "__main__":
    all_paths = os.listdir(TEST_DATASET)
    dset = create_test_dataset(all_paths)
    resnet_loaded =  keras.models.load_model(MODEL_PATH)
    preds_ = resnet_loaded.predict(dset)
    preds = []
    for i in range(len(preds_)):
        preds.append(np.argmax(preds_[i]))

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_paths, preds):
            f.write(f"{name}\t{cl_id}\n")