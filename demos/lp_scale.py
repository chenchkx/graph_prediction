

import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
dir_path = os.path.dirname(__file__)


sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f7"})
from scipy.interpolate import make_interp_spline
plt.axis('equal')  #x轴和y轴等比例
# plt.axis('equal')  #x轴和y轴等比例

np.random.seed(0)

mean = [2.2,2.3]
cov = [[1.212,0.1673],[0.1673,.1288]]
data = np.random.multivariate_normal(mean,cov,300)
w,v = np.linalg.eig(np.array(cov))

x_data = data[:,[0]]
y_data = data[:,[1]]
plt.scatter(x_data,y_data, color=sns.color_palette()[0], marker='o',alpha=0.5,label='Gaussian data') # alpha设置透明度
plt.arrow(0, 0, v[0,0], v[1,0],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'black',ec = 'black')
plt.arrow(0, 0, v[0,1], v[1,1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'black',ec = 'black')


plt.scatter(mean[0], mean[1], color=sns.color_palette()[3],label='Original mean point')
plt.arrow(0, 0, mean[0], mean[1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc =sns.color_palette()[3],ec = sns.color_palette()[3])

new_mean = np.dot(np.array(cov),np.array(mean))
plt.scatter(new_mean[0], new_mean[1], color='coral',label='COV-MAP mean point')
plt.arrow(0, 0, new_mean[0], new_mean[1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'coral',ec = 'coral')




# mean = [2.65,-0.5]
# cov = [[1.212,0.1673],[0.1673,.1288]]
# data = np.random.multivariate_normal(mean,cov,300)
# w,v = np.linalg.eig(np.array(cov))
#
# x_data = data[:,[0]]
# y_data = data[:,[1]]
# plt.scatter(x_data,y_data, color='y', marker='o',alpha=0.5,label='Gaussian data') # alpha设置透明度
# plt.arrow(0, 0, v[0,0], v[1,0],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'black',ec = 'black')
# plt.arrow(0, 0, v[0,1], v[1,1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'black',ec = 'black')
#
#
# plt.scatter(mean[0], mean[1], color='b',label='Original mean point')
# plt.arrow(0, 0, mean[0], mean[1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'coral',ec = 'coral')
#
# new_mean = np.dot(np.array(cov),np.array(mean))
# plt.scatter(new_mean[0], new_mean[1], color='b',label='COV-MAP mean point')
# plt.arrow(0, 0, new_mean[0], new_mean[1],length_includes_head = True,head_width = 0.2,head_length = 0.25,fc = 'maroon',ec = 'maroon')
#

plt.xlim((-1.5, 5.5))
plt.ylim((-1.5, 5.5))


# plt.axis('off')
plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'GaussianInfo.png')
plt.savefig(fig_path)
plt.figure()


plt.show()