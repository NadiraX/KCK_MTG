import os.path
import cv2 as cv
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import colorsys

def gora_dol(imgs,pure_list):
    for i in range(len(imgs)):
        hsv_im = imgs[i].convert('HSV')
        width, height = imgs[i].size

        if width > height:
            imgs[i] = imgs[i].rotate(90)

        h2 = height / 8
        gora = []
        dol = []

        suma1 = 0
        suma2 = 0
        iter1 = 0
        iter2 = 0
        for w in range(width):
            for h in range(height):
                r, g, b = hsv_im.getpixel((w, h))
                v, s, h3 = colorsys.rgb_to_hsv(r, g, b)

                if h > h2*5 and h<h2*7:
                    suma1 += v
                    iter1 += 1
                elif h<h2*3 and h>h2:
                    suma2 += v
                    iter2 += 1
        print(pure_list[i])
        if suma1/iter1 >suma2/iter2:
            print("1")
            pass
        else:
            pass
            print("0")
        imgs[i].show()
def main():
    pure_list = []
    save_path_pure = Path('pure/')
    for filename in os.listdir(save_path_pure):
        pure_list.append(filename)

    imgs = [Image.open(os.path.join(save_path_pure, Path(i))) for i in pure_list]
    # for i in range(len(imgs)):
    #     imgs[i] = imgs[i].convert('L')
    #     imgs[i] = imgs[i].point(lambda x: 0 if x < 128 else 255, '1')
    #     #imgs[i].show()
    gora_dol(imgs,pure_list)

main()