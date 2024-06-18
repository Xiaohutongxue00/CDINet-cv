import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
import os
# Fixing random state for reproducibility
np.random.seed(19680801)
#定义好两条曲线的数据
"""
xdata = np.random.random([2, 10])
xdata1 = xdata[0, :]
xdata2 = xdata[1, :]
xdata1.sort()
xdata2.sort()
ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 3
"""

data_root = 'D:/a_mycollection/SalientDetection'

dataset = 'LFSD'

# \multicolumn{1}{c}{\textbf{A2dele }} &
# \multicolumn{1}{c}{\textbf{JC-DCF  }} &
# \multicolumn{1}{c}{\textbf{SSF   }} &
# \multicolumn{1}{c}{\textbf{CMWNet }} &
# \multicolumn{1}{c}{\textbf{CoNet }} &
# \multicolumn{1}{c}{\textbf{cmMS}} &
# \multicolumn{1}{c}{\textbf{PGAR }} &
# \multicolumn{1}{c}{\textbf{D3Net }} &
# \multicolumn{1}{c}{\textbf{DFNet}} &
# \multicolumn{1}{c}{\textbf{DFMNet }} &
# \multicolumn{1}{c}{\textbf{ICNet}} &
# \multicolumn{1}{c}{\textbf{ASIFNet}} &
# \multicolumn{1}{c}{\textbf{CJLB }} &
# \multicolumn{1}{c}{\textbf{BiANet}} &
# \multicolumn{1}{c}{\textbf{DQSD}} &
# \multicolumn{1}{c}{\textbf{DRLF }} &
# \multicolumn{1}{|c}{\textbf{BMCNet }}

#our model:A2TPNet
p0_path = os.path.join(data_root, 'my_model/BMCNet/P and R', dataset, 'P.npy')
ro_path = os.path.join(data_root, 'my_model/BMCNet/P and R', dataset, 'R.npy')
prec0      = np.load(p0_path)
recall0    = np.load(ro_path)

#A2dele
p1_path = os.path.join(data_root, '2020_CVPR_A2dele/P and R', dataset, 'P.npy')
r1_path = os.path.join(data_root, '2020_CVPR_A2dele/P and R', dataset, 'R.npy')
prec1      = np.load(p1_path)
recall1    = np.load(r1_path)

#JL-DCF
p2_path = os.path.join(data_root, '2020_CVPR_JL-DCF/P and R', dataset, 'P.npy')
r2_path = os.path.join(data_root, '2020_CVPR_JL-DCF/P and R', dataset, 'R.npy')
prec2      = np.load(p2_path)
recall2    = np.load(r2_path)

#SSF
p3_path = os.path.join(data_root, '2020_CVPR_SSF/P and R', dataset, 'P.npy')
r3_path = os.path.join(data_root, '2020_CVPR_SSF/P and R', dataset, 'R.npy')
prec3      = np.load(p3_path)
recall3    = np.load(r3_path)

#CMWNet
p4_path = os.path.join(data_root, '2020_ECCV_CMWNet/P and R', dataset, 'P.npy')
r4_path = os.path.join(data_root, '2020_ECCV_CMWNet/P and R', dataset, 'R.npy')
prec4      = np.load(p4_path)
recall4    = np.load(r4_path)

#CoNet
p5_path = os.path.join(data_root, '2020_ECCV_CoNet/P and R', dataset, 'P.npy')
r5_path = os.path.join(data_root, '2020_ECCV_CoNet/P and R', dataset, 'R.npy')
prec5      = np.load(p5_path)
recall5    = np.load(r5_path)

#cmMS
p6_path = os.path.join(data_root, '2020_ECCV_cmMS/P and R', dataset, 'P.npy')
r6_path = os.path.join(data_root, '2020_ECCV_cmMS/P and R', dataset, 'R.npy')
prec6      = np.load(p6_path)
recall6    = np.load(r6_path)

# PGAR
p7_path = os.path.join(data_root, '2020_ECCV_PGAR/P and R', dataset, 'P.npy')
r7_path = os.path.join(data_root, '2020_ECCV_PGAR/P and R', dataset, 'R.npy')
prec7      = np.load(p7_path)
recall7    = np.load(r7_path)

# D3Net
p8_path = os.path.join(data_root, '2020_TNNLS_D3Net/P and R', dataset, 'P.npy')
r8_path = os.path.join(data_root, '2020_TNNLS_D3Net/P and R', dataset, 'R.npy')
prec8      = np.load(p8_path)
recall8    = np.load(r8_path)

#DFNet
p9_path = os.path.join(data_root, '2020_TIP_DFNet/P and R', dataset, 'P.npy')
r9_path = os.path.join(data_root, '2020_TIP_DFNet/P and R', dataset, 'R.npy')
prec9      = np.load(p9_path)
recall9    = np.load(r9_path)

# DFMNet
p10_path = os.path.join(data_root, '2021_MM_DFMNet/P and R', dataset, 'P.npy')
r10_path = os.path.join(data_root, '2021_MM_DFMNet/P and R', dataset, 'R.npy')
prec10      = np.load(p10_path)
recall10    = np.load(r10_path)

# ICNet
p11_path = os.path.join(data_root, '2020_TIP_ICNet/P and R', dataset, 'P.npy')
r11_path = os.path.join(data_root, '2020_TIP_ICNet/P and R', dataset, 'R.npy')
prec11      = np.load(p11_path)
recall11    = np.load(r11_path)

# ASIFNet
p12_path = os.path.join(data_root, '2021_TCyb_ASIFNet/P and R', dataset, 'P.npy')
r12_path = os.path.join(data_root, '2021_TCyb_ASIFNet/P and R', dataset, 'R.npy')
prec12      = np.load(p12_path)
recall12    = np.load(r12_path)

# CJLB
p13_path = os.path.join(data_root, '2021_NP_CJLB/P and R', dataset, 'P.npy')
r13_path = os.path.join(data_root, '2021_NP_CJLB/P and R', dataset, 'R.npy')
prec13      = np.load(p13_path)
recall13    = np.load(r13_path)


# BIANet
p14_path = os.path.join(data_root, '2021_TIP_BIANet/P and R', dataset, 'P.npy')
r14_path = os.path.join(data_root, '2021_TIP_BIANet/P and R', dataset, 'R.npy')
prec14      = np.load(p14_path)
recall14    = np.load(r14_path)

# DQSD
p15_path = os.path.join(data_root, '2021_TIP_DQSD/P and R', dataset, 'P.npy')
r15_path = os.path.join(data_root, '2021_TIP_DQSD/P and R', dataset, 'R.npy')
prec15      = np.load(p15_path)
recall15    = np.load(r15_path)

# DRLF
p16_path = os.path.join(data_root, '2021_TIP_DRLF/P and R', dataset, 'P.npy')
r16_path = os.path.join(data_root, '2021_TIP_DRLF/P and R', dataset, 'R.npy')
prec16      = np.load(p16_path)
recall16    = np.load(r16_path)

# HAINet
# p17_path = os.path.join(data_root, '2021_TIP_DQSD/P and R', dataset, 'P.npy')
# r17_path = os.path.join(data_root, '2021_TIP_HAINet/P and R', dataset, 'R.npy')
# prec17      = np.load(p17_path)
# recall17    = np.load(r17_path)



# ICNet
# p15_path = os.path.join(data_root, '2020_TIP_ICNet/P and R', dataset, 'P.npy')
# r15_path = os.path.join(data_root, '2020_TIP_ICNet/P and R', dataset, 'R.npy')
# prec15      = np.load(p15_path)
# recall15    = np.load(r15_path)



# CJLB
# p13_path = os.path.join(data_root, '2021_NP_CJLB/P and R', dataset, 'P.npy')
# r13_path = os.path.join(data_root, '2021_NP_CJLB/P and R', dataset, 'R.npy')
# prec13      = np.load(p13_path)
# recall13    = np.load(r13_path)

# DRLF
# p12_path = os.path.join(data_root, '2021_TIP_DRLF/P and R', dataset, 'P.npy')
# r12_path = os.path.join(data_root, '2021_TIP_DRLF/P and R', dataset, 'R.npy')
# prec12      = np.load(p12_path)
# recall12    = np.load(r12_path)


#绘制图画
fig = plt.figure()                         #生成一个画板
ax = fig.add_subplot(1, 1, 1)              #定义有着1行1列子图的视图并取第一个


ax.plot(recall1, prec1, linestyle='--', color='tab:orange', label='A2dele')  #第一条曲线
ax.plot(recall2, prec2, linestyle='--', color='tab:blue', label='JL-DCF')
ax.plot(recall3, prec3, linestyle='--', color='tab:brown', label='SSF')
ax.plot(recall4, prec4, linestyle='--', color='tab:gray', label='CMWNet')
ax.plot(recall5, prec5, linestyle='--', color='yellowgreen', label='CoNet')
ax.plot(recall6, prec6, linestyle='--', color='yellow', label='cmMS')
ax.plot(recall7, prec7, linestyle='--', color='lawngreen', label='PGAR')
ax.plot(recall8, prec8, linestyle='--', color='darkred', label='D3Net')
ax.plot(recall9, prec9, linestyle='--', color='cyan', label='DFNet')
ax.plot(recall10, prec10, linestyle='--', color='sienna', label='DFMNet')
ax.plot(recall11, prec11, linestyle='--', color='deepskyblue', label='ICNet')
ax.plot(recall12, prec12, linestyle='--', color='darkgreen', label='ASIFNet')
ax.plot(recall13, prec13, linestyle='--', color='royalblue', label='CJLB')
ax.plot(recall14, prec14, linestyle='--', color='darkviolet', label='BiANet')
ax.plot(recall15, prec15, linestyle='--', color='pink', label='DQSD')
ax.plot(recall16, prec16, linestyle='--', color='palegreen', label='DRLF')
# ax.plot(recall17, prec17, linestyle='--', color='deeppink', label='HAINet')
ax.plot(recall0, prec0, color='tab:red', label='Ours', linewidth=2)




ax.set_xlim([0, 1])
ax.set_ylim([0.1, 1])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
# 'Ours', 'CPFP', 'TANet', 'A2dele', 'CMWNet', 'DANet', 'HDFNet', 'cmMS', 'DFNet', 'D3Net', 'BiANet', 'DQSD', 'DRLF', 'HAINet', 'ASIF', 'ICNet'
ax.set_title('LFSD', fontsize=20)
# other model salient map\my_model\JALNet\P and R\STERE
plt.grid()
# 'CPFP', 'DMRA', 'MMCI', 'TANet', 'A2dele', 'SSF', 'CMWNet', 'CoNet', 'DANet', 'D3Net', 'DFNet', 'ICNet', 'ASIF', 'CJLB', 'BiANet', 'DQSD', 'HAINet', 'Ours'
plt.legend(('A2dele', 'JL-DCF', 'SSF', 'CMWNet', 'CoNet', 'cmMS', 'PGAR', 'D3Net', 'DFNet', 'D3Net', 'DFNet', 'DFMNet', 'ICNet', 'ASIFNet', 'CJLB', 'BiANet', 'DQSD', 'DRLF', 'Ours'), ncol=2, loc='lower left')
plt.savefig('D:/a_mycollection/SalientDetection/my_model/BMCNet/P and R/LFSD/LFSD_PR.eps')
plt.show()
