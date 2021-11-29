import os

import cv2
import math


def detect(filename):
    # detect the cat face by opencv haarcascade
    # return the coordinates of cat face
    face_cascade = cv2.CascadeClassifier(
        'H://Project/21ACB/cat_face_cut/haarcascade_frontalcatface_extended.xml')
    # read image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.02,
                                          minNeighbors=5)
    size = img.shape[0] * img.shape[1]

    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        w_delta = math.ceil(w * 0.1)
        h_delta = math.ceil(h * 0.1)
        # print(w, h)
        # print(x, y)

        face_size = w * h

        # change the coordinate, increase the rectangle by 10% for each side and move up to include the ears
        x1 = x - w_delta
        y1 = y - h_delta * 2
        x2 = x + w + w_delta
        y2 = y + h

        # draw the cat face, originate and cut
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print(x1, y1, x2, y2)
        # return x1, y1, x2, y2

        coordinates = [x1, y1, x2, y2, w]
        positive = True
        for coordinate in coordinates:
            if coordinate < 0:
                positive = False
                break

        # test whether the detected area is valid
        if w < 30 or (face_size < (size * 0.01)):
            return None

        # return extended box if available
        if positive is True:
            return coordinates
        else:
            coordinates = [x, y, x + w, y + h, w]
            return coordinates


def cut(filename):
    # cut the cat face out according to the coordinates given by detection
    # return the image for saving
    coordinates = detect(filename)
    img = cv2.imread(filename)
    if coordinates is not None:
        new = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
        return new, coordinates
    else:
        return None, None


def recolor(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img
