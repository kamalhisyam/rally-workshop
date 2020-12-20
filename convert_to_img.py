import numpy as np
from PIL import Image


def convert_1D_array_to_img(img_arr, size, fname, mode='RGB'):
    new_img = Image.new(mode=mode, size=size)
    height = size[1]
    i, j = (0, 0)
    for idx in range(len(img_arr)):
        if (i == height):
            i = 0
            j += 1

        r, g, b = img_arr[idx]
        new_img.putpixel((i, j), (r, g, b))
        i += 1

    new_img.save(fname)
