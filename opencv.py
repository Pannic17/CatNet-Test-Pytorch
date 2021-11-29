import cv2
import matplotlib.pyplot as plt

filename = input("Filename:")
path = "H:/Project/21ACB/Test6_mobilenet/test/" + filename

img = cv2.imread(path)
b, g, r = cv2.split(img)
plt_img = cv2.merge([r, g, b])

plt.imshow(plt_img)
plt.show()
