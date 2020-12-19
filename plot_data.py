import numpy as np
import matplotlib.pyplot as plt
from helper import convert_to_training_data

input_array = convert_to_training_data('download.jfif')
output_array = convert_to_training_data('test-img.jpg')

plt.plot(input_array, output_array, 'ro')
plt.show()
