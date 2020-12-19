import tensorflow as tf
import pandas as pd
from IPython import clear_output
from helper import make_input_fn, convert_to_training_data

clear_output()
r_train = convert_to_training_data('test-img.jpg')['R']
df_train = convert_to_training_data('download.jfif')

# extracting the last 10% of sample data to be used as evaluation set
num_of_eval_samples = int(0.1 * r_train.size)
r_eval = r_train[-num_of_eval_samples:]
r_train = r_train.iloc[:-num_of_eval_samples]
df_eval = df_train[-num_of_eval_samples:]
df_train = df_train[:-num_of_eval_samples]

# populating the feature_columns
feature_columns = []
for feature_name in df_train.columns:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float64))

# here we will call the input_function that was returned to us to get a dataset object we can feed to the model
train_input_fn = make_input_fn(df_train, r_train)
eval_input_fn = make_input_fn(df_eval, r_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
# get model metrics/stats by testing on tetsing data

pred_dicts = list(linear_est.predict(eval_input_fn))
predicted_output = pd.Series([pred['predictions'][0] for pred in pred_dicts])

# the result variable is simply a dict of stats about our model
print('Predicted output:')
print(predicted_output.head())
tf.saved_model.save(linear_est, '/')
