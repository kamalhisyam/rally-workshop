import numpy as np
import pandas as pd
from helper import convert_to_training_data

a = convert_to_training_data('download.jfif')
r_train = convert_to_training_data('test-img.jpg').pop('R')

print(a.head())
print(type(r_train))
