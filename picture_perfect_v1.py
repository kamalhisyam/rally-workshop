import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from helper import make_input_fn, convert_to_training_data
from convert_to_img import convert_1D_array_to_img


rgb_train = convert_to_training_data('after_vandy.jpg')
r_train = rgb_train['R']
g_train = rgb_train['G']
b_train = rgb_train['B']
df_train = convert_to_training_data('before_vandy.jpg')

# extracting the last 10% of sample data to be used as evaluation set
num_of_eval_samples = int(0.1 * rgb_train.size)
r_eval = r_train[-num_of_eval_samples:]
r_train = r_train.iloc[:-num_of_eval_samples]
g_eval = g_train[-num_of_eval_samples:]
g_train = g_train.iloc[:-num_of_eval_samples]
b_eval = b_train[-num_of_eval_samples:]
b_train = b_train.iloc[:-num_of_eval_samples]
df_eval = df_train[-num_of_eval_samples:]
df_train = df_train[:-num_of_eval_samples]

# populating the feature_columns
feature_columns = []
for feature_name in df_train.columns:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float64))

# here we will call the input_function that was returned to us to get a dataset object we can feed to the model
r_train_input_fn = make_input_fn(df_train, r_train)
g_train_input_fn = make_input_fn(df_train, g_train)
b_train_input_fn = make_input_fn(df_train, b_train)
eval_input_fn = make_input_fn(df_eval, r_eval, num_epochs=1, shuffle=False)

# creating linear estimator for each of R, G, B channel
r_linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
g_linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
b_linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# training models
r_linear_est.train(r_train_input_fn)
g_linear_est.train(g_train_input_fn)
b_linear_est.train(b_train_input_fn)

# pred_dicts = list(linear_est.predict(eval_input_fn))
# predicted_output = pd.Series([pred['predictions'][0] for pred in pred_dicts])

# Testing

test_img = Image.open('before.png')

df_test = pd.DataFrame(np.row_stack(np.asarray(
    test_img) / 255.0), columns=['R', 'G', 'B'])

r_pred_input_fn = make_input_fn(
    df_test, df_test['R'], num_epochs=1, shuffle=False)

g_pred_input_fn = make_input_fn(
    df_test, df_test['G'], num_epochs=1, shuffle=False)

b_pred_input_fn = make_input_fn(
    df_test, df_test['B'], num_epochs=1, shuffle=False)

r_pred_dicts = list(r_linear_est.predict(input_fn=r_pred_input_fn))
g_pred_dicts = list(g_linear_est.predict(input_fn=g_pred_input_fn))
b_pred_dicts = list(b_linear_est.predict(input_fn=b_pred_input_fn))

r_pred_arr = [pred['predictions'][0] for pred in r_pred_dicts]
g_pred_arr = [pred['predictions'][0] for pred in g_pred_dicts]
b_pred_arr = [pred['predictions'][0] for pred in b_pred_dicts]

rgb_pred_arr = np.empty((len(r_pred_arr), 3))
for i in range(len(rgb_pred_arr)):
    rgb_pred_arr[i] = [r_pred_arr[i], g_pred_arr[i], b_pred_arr[i]]

rgb_pred_arr = rgb_pred_arr * 255

convert_1D_array_to_img(rgb_pred_arr.astype(
    int, copy=False), (256, 256), fname='output1-img.jpg')
