import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

img = Image.open('download.jfif')
out_img = Image.new('RGB', img.size, 0xffffff)

width, height = img.size
for i in range(width):
    for j in range(height):
        r, g, b = img.getpixel((i, j))
        if (b > g):
            out_img.putpixel((i, j), (r, g, b))

out_img.save('test-img.jpg')
