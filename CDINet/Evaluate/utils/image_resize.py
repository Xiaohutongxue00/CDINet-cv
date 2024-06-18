import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

mask_path = 'E:/My work and studay/my papaer work/MyCollectDatas/SalientDetection/evalute/STERE/Mask'
mask_resize_path = 'E:/My work and studay/my papaer work/MyCollectDatas/SalientDetection/evalute/STERE/Mask_224'
print(mask_path)
for file in os.listdir(mask_path):
    print(file)
    file_path = os.path.join(mask_path, file)
    print(file_path)
    im1 = cv2.imread(file_path)

    im2 = cv2.resize(im1, (224, 224))
    cv2.imwrite(os.path.join(mask_resize_path, file), im2)
