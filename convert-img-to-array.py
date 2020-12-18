import numpy as np
from PIL import Image, ImageOps

img = Image.open('download.jfif')
img_width, img_height = img.size
img_array = np.asarray(img)

# initialize 1D array
new_array = np.empty((img_width + img_height, 3))

# converting to img_array to 1D array
for i in range(img_width):
    for j in range(img_height):
        new_array[i+j] = img_array[i][j]


print(new_array)
