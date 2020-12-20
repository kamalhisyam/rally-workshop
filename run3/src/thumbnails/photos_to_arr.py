# This script generates a .txt file containing a row-stacked RGB values of all .jpg files in the directory

from PIL import Image
import glob
import os

size = 128, 128
foutput = 'inputs.txt'

for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    os.chdir('thumbnails')
    im.save(file + "_thumbnail" + ".jpg")
    os.chdir(os.pardir)
