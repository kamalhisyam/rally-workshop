# This script converts all jpg file in current dir to thumbnail size and save the files in 'thumbnails' folder

from PIL import Image
import glob
import os

size = 128, 128

if not os.path.exists('thumbnails'):
    os.makedirs('thumbnails')

for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    os.chdir('thumbnails')
    im.save(file + "_thumbnail" + ".jpg")
    os.chdir(os.pardir)
