import numpy as np
import os
import pydensecrf.densecrf as dcrf

import math

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import PIL
import numpy as np
import os
import torch
import cv2

def crf_refine(img, annos):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    print(img.shape[:2],annos.shape)
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32') # 创建和输入图片同样大小的U
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])  # 和输入图片同样大小
    return res.astype('uint8')

if __name__ == "__main__":
    # path_imgae = 'E:/dataset/RGB-D_Dataset/SIP/Image'
    dataset = 'NJU2K'

    path_imgae = os.path.join('D:/a_mycollection/SalientDetection/evalute/test_data', dataset, 'RGB')
    # path_imgae = "D:/a_mycollection/SalientDetection/evalute/test_data/RGBD135/RGB"
    # root1_path = 'E:/My work and studay/my papaer work/Dense Alternate  Fusion Network with multi-modal attention for asymmetical RGB-D Salient Object Detection/'
    # root2_path = 'code/modified_5_version/DHAFNet-master/output/DHAFNet_VGG16_7Datasets/pre/SIP'
    root1_path = os.path.join('E:/My work and studay/my papaer work/M2AS/result/version_10/epoch_120', dataset)
    # root1_path = "E:/My work and studay/my papaer work/CWMI-EI/result/version_9/epoch_130/RGBD135"
    root2_path = os.path.join('E:/My work and studay/my papaer work/M2AS/result/version_10/epoch_120/refinment', dataset)
    # root2_path = "E:/My work and studay/my papaer work/CWMI-EI/result/version_9/epoch_130/refinment/RGBD135"
    path_mask = root1_path
    path_save = root2_path
    # path_mask = 'E:/My work and studay/my papaer work/Depth-ware and hierarchical alternate fusion network for RGB-D Salient object detection/RGBD_SOD_model/modified_12_version/DHAFNet-master/output/DHAFNet_VGG16_7Datasets/pre/NLPR'
    # path_save = 'E:/My work and studay/my papaer work/Depth-ware and hierarchical alternate fusion network for RGB-D Salient object detection/RGBD_SOD_model/modified_12_version/DHAFNet-master/output/DHAFNet_VGG16_7Datasets/pre/refinement/NLPR'
    # img_list = os.listdir(path_mask)
    to_pil = transforms.ToPILImage()
    #
    # tqdm_iter = tqdm(enumerate(img_list), total=len(img_list), leave=False)
    img_list = sorted(os.listdir(path_mask))
    for file in tqdm(img_list, total=len(img_list)):
        file_name = str(file).split('.')[0]
        image = Image.open(os.path.join(path_imgae, file_name + '.jpg')).convert("RGB")
        mask = cv2.imread(os.path.join(path_mask, file), 0)
        prediction = crf_refine(np.array(image), np.array(mask))
        pre_mask = to_pil(prediction)
        pre_mask.save(os.path.join(path_save, file))