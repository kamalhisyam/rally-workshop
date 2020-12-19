import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps


def convert_to_training_data(img_name):
    '''Takes an image file and converts it into normalized 1D array. Assumes RGB color system'''
    img = Image.open(img_name)
    img_width, img_height = img.size
    img_array = np.asarray(img)

    # initialize 1D array
    new_array = np.empty((img_width * img_height, 3))
    idx = 0

    # converting to img_array to 1D array [x / 255 for x in img_array[i][j]]

    for i in range(img_width):
        for j in range(img_height):
            new_array[idx] = img_array[i][j] / 255.0
            idx += 1

    return(pd.DataFrame(np.row_stack(new_array), columns=['R', 'G', 'B']))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use
