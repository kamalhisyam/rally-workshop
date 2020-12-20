import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from helper.helper import make_input_fn
from helper.img_to_df import img_to_df
from convert_to_img import convert_1D_array_to_img
from train_model import train_channel, pred_channel

# Filenames
LABEL_CSV = 'after.csv'
FEATURES_CSV = 'before.csv'
TEST_IN_FNAME = 'test2_before.jpg'
TEST_OUT_FNAME = 'test4_after_pred.jpg'

rgb_train = pd.read_csv(LABEL_CSV)
df_train = pd.read_csv(FEATURES_CSV)

# Populating the feature_columns
feature_cols = [tf.feature_column.numeric_column(
    ft, dtype=tf.float64) for ft in df_train.columns]

# Call the input_function that was returned to us to get a dataset object we can feed to the model
r_linear_est = train_channel(rgb_train['R'], df_train, feature_cols)
g_linear_est = train_channel(rgb_train['G'], df_train, feature_cols)
b_linear_est = train_channel(rgb_train['B'], df_train, feature_cols)

# Prediction
test_img = Image.open(TEST_IN_FNAME)
df_test = img_to_df(TEST_IN_FNAME)

r_pred_arr = pred_channel(df_test, r_linear_est, 'R')
g_pred_arr = pred_channel(df_test, r_linear_est, 'G')
b_pred_arr = pred_channel(df_test, r_linear_est, 'B')

rgb_pred_arr = np.empty((len(r_pred_arr), 3))
for i in range(len(rgb_pred_arr)):
    rgb_pred_arr[i] = [r_pred_arr[i], g_pred_arr[i], b_pred_arr[i]] * 255

convert_1D_array_to_img(img_arr=img_arr.astype(int), size=test_img.size,
                        fname=TEST_OUT_FNAME)
