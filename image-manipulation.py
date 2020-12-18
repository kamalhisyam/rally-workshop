import numpy as np
from PIL import Image, ImageOps

img = Image.open('download.jfif')
out_img = Image.new('RGB', img.size, 0xffffff)

width, height = img.size
for i in range(width):
    for j in range(height):
        r, g, b = img.getpixel((i, j))
        img_intensity = int((r + g + b)/3)
        out_img.putpixel((i, j), (img_intensity, img_intensity, img_intensity))


out_img.save('test-img.jpg')
