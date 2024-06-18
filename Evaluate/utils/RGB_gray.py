import cv2
import numpy as np
import os
from torchvision import transforms

lenna = cv2.imread("E:/image/11.png")
row, col, channel = lenna.shape
lenna_gray = np.zeros((row, col))
for r in range(row):
    for l in range(col):
        lenna_gray[r, l] = 1 / 3 * lenna[r, l, 0] + 1 / 3 * lenna[r, l, 1] + 1 / 3 * lenna[r, l, 2]

cv2.imwrite("E:/image/image11.png", lenna_gray)
# cv2.imshow("lenna_gray", lenna_gray.astype("uint8"))
# cv2.waitKey()