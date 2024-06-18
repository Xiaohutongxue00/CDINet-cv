import os
import numpy
import torch

test_path = 'E:/My work and studay/my papaer work/other model salient map/my_model/JALNet/NLPR'
gt_path = 'E:/My work and studay/my papaer work/other model salient map/evalute/NLPR/Mask_256/'

img = []
for image in os.listdir(test_path):
    img.append(image)

for file in os.listdir(gt_path):
    if file not in img:
        delete_path = os.path.join(gt_path, file)
        os.remove(delete_path)
        print(file)
    # else:
    #     print(file)
