import os
import cv2 as cv
data_root = "E:/My work and studay/my papaer work/MyCollectDatas/SalientDetection"
mask_root = os.path.join(data_root, "2020_ECCV_cmMS/results")
# test_datasets = ["NLPR", 'SSD', 'RGBD135', 'DUT', 'SIP', 'STERE', 'LFSD', 'NJU2K']
pred_root = os.path.join(data_root, "2020_CVPR_JL-DCF/SaliencyMap/STERE")
data_path = pred_root
for file in os.listdir(data_path):

    #图片名称为3
    # name1 = file.split("_")[0]
    # name2 = file.split("_")[1]
    # name3 = file.split("_")[2]
    # name = name1+'_'+name2 + '_'+name3+ '.png'

    # 图片名称为2
    # name1 = file.split(".")[0]
    # name2 = name1.split("_")[0]
    # name3 = name1.split("_")[1]
    # name = name2 + '_'+ name3 + '.png'

    # 图片名称为1
    name = file.split(".")[0]
    name = name + '.png'

    old_name = os.path.join(data_path, file)
    new_name = os.path.join(data_path, name)
    print(f"之前的名称： {old_name}")
    print(f"之后的名称：{new_name}")
    os.renames(old_name, new_name)



