import numpy as np


def convert_1D_array_to_img(img_arr, shape):
    new_arr = np.empty(shape, 'int')
    width, height = shape
    i, j = (0, 0)
    for idx in range(len(img_arr)):
        if (i == width):
            i = 0
            j += 1

        new_arr[i][j] = img_arr[idx]
        i += 1

    return new_arr
