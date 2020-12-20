import tensorflow as tf
from helper.helper import make_input_fn
from helper.img_to_df import img_to_df


def train_channel(label_train, df_train, feature_columns):

    linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    linear_est.train(make_input_fn(df_train, label_train))

    return linear_est


def pred_channel_output(df_test, linear_est, channel):

    pred_input_fn = make_input_fn(
        df_test, df_test[channel], num_epochs=1, shuffle=False)

    pred_dicts = list(linear_est.predict(input_fn=pred_input_fn))

    pred_arr = [pred['predictions'][0] for pred in pred_dicts]

    return pred_arr
