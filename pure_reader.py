import os.path
import cv2 as cv
from pathlib import Path
from PIL import Image
import colorsys


def main():
    pure_list = []
    save_path_pure = Path('pure/')
    for filename in os.listdir(save_path_pure):
        pure_list.append(filename)

    imgs = [Image.open(os.path.join(save_path_pure, Path(i))) for i in pure_list]
    hsv_im = imgs[0].convert('HSV')
    width, height = imgs[0].size

    h2 = height/2
    gora =[]
    dol = []
    for w in range(width):
        for h in range(height):
            r,g,b = hsv_im.getpixel((w,h))
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            if h >h2:
                if v not in gora:
                    gora.append(v)
            else:
                if v not in dol:
                    dol.append(v)
    print(len(gora))
    print(len(dol))
    print(dol)
main()