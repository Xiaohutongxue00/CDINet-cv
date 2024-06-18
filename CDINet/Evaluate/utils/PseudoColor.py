import cv2
import numpy as np
import os
from tqdm import tqdm

gt_root = "D:/a_mycollection/SalientDetection/evalute/test_data"
pre_root = "E:/My work and studay/my papaer work/M2AS/result/version_10/epoch_120/output1"
out_root = "E:/My work and studay/my papaer work/M2AS/result/version_10/epoch_120/output1/color"
dataset = 'LFSD'
out_root = os.path.join(out_root, dataset)

gt_path = os.path.join(gt_root, dataset, 'GT')
pre_path = os.path.join(pre_root, dataset)
print(f"--------------测试数据集{dataset}----------------")
mask_name_list = sorted(os.listdir(pre_path))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(gt_path, mask_name)
    pred_path = os.path.join(pre_path, mask_name)
    out_path = os.path.join(out_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    pred = np.array(pred)
    pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

    # color_zero = np.zeros(shape = pred.shape).astype(np.uint8)
    # color_zero[0:50,0:50] = 254
    # color_img = color_zero
    # color_gray = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    # color_app = cv2.applyColorMap(color_gray,2)
    # out = cv2.addWeighted(pred,0.5,color_app,0.5,0)
    cv2.imwrite(out_path,pred)
