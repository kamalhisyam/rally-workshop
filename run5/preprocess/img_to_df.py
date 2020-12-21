import numpy as np
import pandas as pd
from PIL import Image


def img_to_df(img_name):
    '''Takes an image file and converts it into normalized 1D array. Assumes RGB color system'''
    img = Image.open(img_name)
    img_width, img_height = img.size
    img_array = np.asarray(img)

    # initialize 1D array
    new_array = np.empty((img_width * img_height, 5))
    idx = 0

    for i in range(img_height):
        for j in range(img_width):
            r, g, b = img_array[i][j] / 255.0
            new_array[idx] = [r, g, b, (i + 1)/img_height, (j+1)/img_width]
            idx += 1

    return(pd.DataFrame(np.row_stack(new_array), columns=['R', 'G', 'B', 'X', 'Y']))
