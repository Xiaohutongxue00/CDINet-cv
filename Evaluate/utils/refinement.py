import math
from utils import crf_refine

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import PIL
import numpy as np
import os
import torch
import cv2

def get_mask(num, threshold):
    num = num/255
    threshold = threshold/255
    if num < threshold:
        y = 10*(num/threshold -1)
    else:
        y = 10*((num - threshold)/(1-threshold))
    result = 0.5* (num + 1/(1 + math.exp(-y)))
    return math.ceil(result*255)

def refin_mask(path, path_save):
    name = os.path.basename(path_save)
    img = cv2.imread(path, 0)
    height, width = img.shape
    to_pil = transforms.ToPILImage()
    img_f = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            t = img[i][j]
            threshold = 150
            # if t > threshold:
            #     img_f[i][j] = 255
            # # elif t<120:
            # #     img_f[i][j] = 0
            # else:
            #     img_f[i][j] = 0
            img_f[i][j] = get_mask(t, threshold)
    pre_mask = to_pil(img_f)
    pre_mask.save(path_save)
    print("{}处理完成!".format(name))

def refinement(path, path_save):
    name = os.path.basename(path_save)
    img = cv2.imread(path, 0)
    height, width = img.shape
    to_pil = transforms.ToPILImage()
    img_f = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            t = img[i][j]
            if t > 150:
                img_f[i][j] = 255
            # elif t<120:
            #     img_f[i][j] = 0
            else:
                img_f[i][j] = 0
    pre_mask = to_pil(img_f)
    pre_mask.save(path_save)
    print("{}处理完成!".format(name))

path_imgae = 'E:/dataset/RGB-D_Dataset/LFSD/Image'
path_mask = 'E:/My work and studay/my papaer work/other model salient map/my_model/LFSD/'
path_save = 'E:/My work and studay/my papaer work/other model salient map/my_model/midified1/LFSD/'
img_list=[]
to_pil = transforms.ToPILImage()
for file in os.listdir(path_mask):
    img_list.append(file)
tqdm_iter = tqdm(enumerate(img_list), total=len(img_list), leave=False)
for index, file in tqdm_iter:
    file_name = str(file).split('.')[0]
    image = Image.open(os.path.join(path_imgae, file_name + '.jpg')).convert("RGB")
    mask = cv2.imread(os.path.join(path_mask, file), 0)
    prediction = crf_refine(np.array(image), np.array(mask))
    pre_mask = to_pil(prediction)
    pre_mask.save(os.path.join(path_save, file))

