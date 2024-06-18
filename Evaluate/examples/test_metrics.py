# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


S = "DUR-OMRON"

FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()
test_datasets = ['LFSD', 'NJU2K', 'NLPR', 'SSD', 'RGBD135', 'DUT', 'SIP', 'STERE']
# 'LFSD', 'NJU2K', "NLPR", 'SSD', 'RGBD135', 'DUT', 'SIP', 'STERE'
# gt_root = "/mnt/harddisk1/gaomengya/codespace/TestDataset/"
gt_root  = "../../dataset/CDINet_test_data/"
# pre_root = "E:/My work and studay/my papaer work/BMCNet/result/version_4"

pre_root = "../../saliency_maps/"

all_result = []
for dataset in test_datasets:
    gt_path = os.path.join(gt_root, dataset, 'GT')
    pre_path = os.path.join(pre_root, dataset)
    print(f"--------------测试数据集{dataset}----------------")
    mask_name_list = sorted(os.listdir(pre_path))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(gt_path, mask_name)
        pred_path = os.path.join(pre_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    pr = FM.get_results()["pr"]

    results = {
        'dataset_name': dataset,
        "adpEm": em["adp"],
        "maxEm": em["curve"].max(),
        "meanEm": em["curve"].mean(),
        "Smeasure": sm,
        "adpFm": fm["adp"],
        "maxFm": fm["curve"].max(),
        "meanFm": fm["curve"].mean(),
        "wFmeasure": wfm,
        "MAE": mae,
        # "precision.shape": pr["p"].shape,
        # "recall.shape": pr["r"].shape,
    }
    all_result.append(results)
    precision = pr["p"]
    recall = pr["r"]
    # E:\My work and studay\my papaer work\other model salient map\my_model\JALNet\P and R
    # p_root = os.path.join(data_root, "2018-Cybernetics-CTMF/P and R/RGBD135/p.npy")
    # r_root = os.path.join(data_root, "2018-Cybernetics-CTMF/P and R/RGBD135/R.npy")
    # np.save(p_root, precision)
    # np.save(r_root, recall)
    # print(results)
    for key, value in results.items():
        print("{}:{}".format(key, value))

with open(os.path.join(pre_path, 'test_dataset_result.txt'),'w') as f:
    f.write(all_result)
print(all_result)

