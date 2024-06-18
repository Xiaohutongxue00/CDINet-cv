import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np
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
prec0      = np.load('')
recall0    = np.load('')

prec1      = np.load('')
recall1    = np.load('')

prec2      = np.load('')
recall2    = np.load('')

prec3      = np.load('')
recall3    = np.load('')

prec4      = np.load('')
recall4    = np.load('')


#绘制图画
fig = plt.figure()                         #生成一个画板
ax = fig.add_subplot(1, 1, 1)              #定义有着1行1列子图的视图并取第一个

ax.plot(recall0,prec0,color='tab:red',label = 'Our',linewidth=2)
#ax.plot(recall1, prec1, linestyle='--',color='tab:orange',label = 'MINet')  #第一条曲线
#ax.plot(recall2,prec2, linestyle='--',color='tab:blue',label = 'GateNet')
#ax.plot(recall3,prec3,linestyle='--',color='tab:brown',label = 'U2-Net')
ax.plot(recall4,prec4,color='tab:gray',label = 'R2Net')



ax.set_xlim([0, 1])
ax.set_ylim([0.3, 1])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')

ax.set_title('SOD',fontsize=20)
plt.savefig('')
plt.show()
