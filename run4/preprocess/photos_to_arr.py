# This script generates a .txt file containing a row-stacked RGB values of all .jpg files in the directory

import numpy as np
from PIL import Image
from img_to_df import img_to_df
import glob
import os
import pandas as pd

ls_df = []

for infile in glob.glob("*after_thumbnail.jpg"):
    file, ext = os.path.splitext(infile)
    ls_df.append(img_to_df(infile))

output = pd.concat(ls_df)
output.to_csv('after.csv')
