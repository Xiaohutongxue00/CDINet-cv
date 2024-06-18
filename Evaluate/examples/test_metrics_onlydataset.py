# # -*- coding: utf-8 -*-
# # @Time    : 2020/11/21
# # @Author  : Lart Pang
# # @GitHub  : https://github.com/lartpang
#
# import os
#
# import cv2
# import numpy as np
# import torch.nn
# from tqdm import tqdm
# from PIL import Image
# import torch.nn.functional as F
# import sys
# sys.path.append('../')
#
# # pip install pysodmetrics
# from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
#
# FM = Fmeasure()
# WFM = WeightedFmeasure()
# SM = Smeasure()
# EM = Emeasure()
# MAE = MAE()
# # test_datasets = ["NLPR", 'SSD', 'RGBD135', 'DUT', 'SIP', 'STERE', 'LFSD', 'NJU2K']
# # E:\My work and studay\my papaer work\MDFNet\result\ablation\baseline
# gt_root  = "/mnt/datasets/rgbdsod/dataset/test_data"
# pre_root = "/mnt/harddisk1/Yangfeng/codespace/DQWNet1/saliency_maps/"
# # pre_root = "D:/a_mycollection/SalientDetection/2021_CVPR_DSA2F/SaliencyMap"
#
#
# # "E:/My work and studay/my papaer work/BMCNet/result/version_7/refinment"
# # pre_root = "E:/My work and studay/my papaer work/MyCollectDatas/SalientDetection/2021_MM_CDINet/SaliencyMap"
# # mask_root = os.path.join(data_root, "evalute/NJUD/Mask")
# # G:/paper/Saliency Detection/2021/ACMM/Cross-modality Discrepant Interaction Network for RGB-D/Mytest_saliencyMaps/SIP
# # E:/My work and studay/my papaer work\BMCNet/result/version_3/STERE
# # pred_root = os.path.join(data_root, "E:/My work and studay/my papaer work\BMCNet/result/version_4/epoch100/NJU2K")
# # all_result = []
# # for dataset in test_datasets:
# dataset = 'NJU2K'
# gt_path = os.path.join(gt_root, dataset, 'GT')
# pre_path = os.path.join(pre_root, dataset)
# print(f"--------------测试数据集{dataset}----------------")
# mask_name_list = sorted(os.listdir(pre_path))
# for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
#     mask_path = os.path.join(gt_path, mask_name)
#     pred_path = os.path.join(pre_path, mask_name)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#     print("cv data.shape = {}".format(type(pred)))
#     FM.step(pred=pred, gt=mask)
#     WFM.step(pred=pred, gt=mask)
#     SM.step(pred=pred, gt=mask)
#     EM.step(pred=pred, gt=mask)
#     MAE.step(pred=pred, gt=mask)
#
# fm = FM.get_results()["fm"]
# wfm = WFM.get_results()["wfm"]
# sm = SM.get_results()["sm"]
# em = EM.get_results()["em"]
# mae = MAE.get_results()["mae"]
# pr = FM.get_results()["pr"]
#
# results = {
#     # 'dataset_name': dataset,
#     "adpEm": em["adp"],
#     "maxEm": em["curve"].max(),
#     "meanEm": em["curve"].mean(),
#     "Smeasure": sm,
#     "adpFm": fm["adp"],
#     "maxFm": fm["curve"].max(),
#     "meanFm": fm["curve"].mean(),
#     "wFmeasure": wfm,
#     "MAE": mae,
#     "precision.shape": pr["p"].shape,
#     "recall.shape": pr["r"].shape,
# }
# # all_result.append(results)
# precision = pr["p"]
# recall = pr["r"]
# # D:/a_mycollection/SalientDetection/my_model/BMCNet/P and R/LFSD
# # # E:\My work and studay\my papaer work\other model salient map\my_model\JALNet\P and R
# # data_root = "D:/a_mycollection/SalientDetection"
# # p_root = os.path.join(data_root, "2021_CVPR_DSA2F/P and R", dataset, "p.npy")
# # r_root = os.path.join(data_root, "2021_CVPR_DSA2F/P and R", dataset, "R.npy")
# # np.save(p_root, precision)
# # np.save(r_root, recall)
# # print(results)
# for key, value in results.items():
#     # print("{}:{}".format(key, value))
#
#
# # for key, value in results.items():
# #
# #     value = float(value)
#
#     # print(f"value.shape = {type(float(value))}")
#     print(round(value, 3))
#
#
#
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import cv2
import numpy as np
import torch.nn
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('../')

# pip install pysodmetrics
from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()
# test_datasets = ["NLPR", 'SSD', 'RGBD135', 'DUT', 'SIP', 'STERE', 'LFSD', 'NJU2K']
# E:\My work and studay\my papaer work\MDFNet\result\ablation\baseline
gt_root  = "./dataset/CDINet_test_data/"
pre_root ="./saliency_maps/"
# pre_root = "D:/a_mycollection/SalientDetection/2021_CVPR_DSA2F/SaliencyMap"


# "E:/My work and studay/my papaer work/BMCNet/result/version_7/refinment"
# pre_root = "E:/My work and studay/my papaer work/MyCollectDatas/SalientDetection/2021_MM_CDINet/SaliencyMap"
# mask_root = os.path.join(data_root, "evalute/NJUD/Mask")
# G:/paper/Saliency Detection/2021/ACMM/Cross-modality Discrepant Interaction Network for RGB-D/Mytest_saliencyMaps/SIP
# E:/My work and studay/my papaer work\BMCNet/result/version_3/STERE
# pred_root = os.path.join(data_root, "E:/My work and studay/my papaer work\BMCNet/result/version_4/epoch100/NJU2K")
# all_result = []
# for dataset in test_datasets:
dataset = ['NJU2K','NLPR', 'LFSD', 'SSD','SIP', 'STERE','DUT','RGBD135'] 
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
    # 'dataset_name': dataset,
    "adpEm": em["adp"],
    "maxEm": em["curve"].max(),
    "meanEm": em["curve"].mean(),
    "Smeasure": sm,
    "adpFm": fm["adp"],
    "maxFm": fm["curve"].max(),
    "meanFm": fm["curve"].mean(),
    "wFmeasure": wfm,
    "MAE": mae,
    "precision.shape": pr["p"].shape,
    "recall.shape": pr["r"].shape,
}
# all_result.append(results)
precision = pr["p"]
recall = pr["r"]
# print("precision.shape = {}".format(type(precision)))
# D:/a_mycollection/SalientDetection/my_model/BMCNet/P and R/LFSD
# E:\My work and studay\my papaer work\other model salient map\my_model\JALNet\P and R
#__________________________________________________________________
# data_root = "E:/SalientDetection/DQWNet/"
# p_root = os.path.join(data_root, "P and R", dataset, "p.npy")
# r_root = os.path.join(data_root, "P and R", dataset, "R.npy")
# np.save(p_root, precision)
# np.save(r_root, recall)
#__________________________________________________________________

# print(results)
for key, value in results.items():
    # print("{}:{}".format(key, value))


# for key, value in results.items():
#
#     value = float(value)

    # print(f"value.shape = {type(float(value))}")
    print(round(value, 3))



