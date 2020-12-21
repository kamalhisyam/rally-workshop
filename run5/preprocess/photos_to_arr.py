# This script generates a .txt file containing a row-stacked RGB values of all .jpg files in the directory

import numpy as np
from PIL import Image
from img_to_df import img_to_df
import glob
import os
import pandas as pd

BEFORE_AFTER = 'before'

ls_df = [img_to_df(infile) for infile in glob.glob(
    "*" + BEFORE_AFTER + "_thumbnail.jpg")]

output = pd.concat(ls_df)
output.to_csv(BEFORE_AFTER + ".csv")
