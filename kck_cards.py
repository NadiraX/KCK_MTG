import numpy as np
import cv2 as cv
import random as rng
import matplotlib.pyplot as plt
from pathlib import Path
import os.path
from PIL import Image
import PIL


def resizeImage(image, size, proportional):
    if proportional:
        h, w = image.shape[:2]
        factor = min(size[0] / h, size[1] / w)
        return cv.resize(image, None, fx = factor, fy = factor)
    return cv.resize(image, None, size)

def drawContours(image, contours):
    for i in range(0, len(contours)):
        color = (255, 255, 255)
        cv.drawContours(image, contours, i, color, -1)
        #cv.fillPoly(image, pts=[contours], color=(255, 255, 255))
        M = cv.moments(contours[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(image, (cx, cy), 2, (255, 255, 255), 2)
    return image

def getContours(image):
    _, thresh = cv.threshold(image, 127, 255, 0) #127
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours

def thresholding(image):
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 1551, 25)
    image = cv.medianBlur(image, 3)
    return image

def alterColors(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image[:,:,1] = 0
    image[:,:,0] = 0
    image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.clip(image, 0, 148)
    return image

def morphImage(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv.dilate(image, kernel, iterations = 2) #2
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return image

def processImage(image):
    image = resizeImage(image, (1200,  1200), True)
    gray = alterColors(image)
    thresh = thresholding(gray)
    thresh_plus = morphImage(thresh)
    contours = getContours(thresh_plus)
    image = drawContours(image, contours)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image,kernel,iterations=6)
    image = resizeImage(image, (800,  600), True)
    return image

def main():
    images = []

    save_path = Path('data/')
    nazwa = 'card'
    end = 'jpg'
    for i in range(21):
        if i < 10:
            images.append(os.path.join(save_path,Path(nazwa + '0' + str(i) +'.' + end)))
        else:
            images.append(os.path.join(save_path,Path(nazwa + str(i) + '.'+ end)))
    for i in range(len(images)-2):
        images.pop()

    images_zmienione = []
    for img in images:
        images_zmienione.append(img[:-4] + '_zmienione' + '.jpg')

    for i in range(len(images)):
        image = cv.imread(images[i])
        image = processImage(image)
        print(images_zmienione)
        cv.imwrite(images_zmienione[i],image)



    imgs = [PIL.Image.open(i) for i in images_zmienione]

    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save('all_cards.jpg')
main()