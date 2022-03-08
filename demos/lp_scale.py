

import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
dir_path = os.path.dirname(__file__)


sns.set()
sns.set_style("darkgrid", {"axes.facecolor": "#f0f0f7"})
from scipy.interpolate import make_interp_spline
# plt.axis('equal')  #x轴和y轴等比例/
# plt.axis('equal')  #x轴和y轴等比例

np.random.seed(0)

mean1 = [2.5, 2.7]
cov1 = [[1.212,0.1673],[0.1673,.1288]]
data1 = np.random.multivariate_normal(mean1,cov1,100)

x_data1 = data1[:,[0]]
y_data1 = data1[:,[1]]
plt.scatter(x_data1, y_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度

mean2 = [-3.2, -2.3]
cov2 = [[1.912,0.5],[0.5,1.1288]]
data2 = np.random.multivariate_normal(mean2,cov2,200)

x_data2 = data2[:,[0]]
y_data2 = data2[:,[1]]
plt.scatter(x_data2, y_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

mean3 = [0.2, -0.4]
cov3 = [[0.212,0.03],[0.03,0.0888]]
data3 = np.random.multivariate_normal(mean3,cov3,60)

x_data3 = data3[:,[0]]
y_data3 = data3[:,[1]]
plt.scatter(x_data3, y_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度

mean4 = [-3.2, 2.3]
cov4 = [[1.812,0.13],[0.13,1.888]]
data4 = np.random.multivariate_normal(mean4,cov4,120)

x_data4 = data4[:,[0]]
y_data4 = data4[:,[1]]
plt.scatter(x_data4, y_data4, color=sns.color_palette()[3], marker='.',alpha=0.9) # alpha设置透明度

total_data = np.concatenate((data1, data2, data3, data4), axis=0)
total_mean = np.mean(total_data, axis=0)
total_std = np.std(total_data, axis=0)
plt.scatter(total_mean[0], total_mean[1], color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.xlim((-6, 6))
plt.ylim((-6, 6))

# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'origin_distribution.png')
plt.savefig(fig_path)
plt.figure()
plt.show()


### generate bn data

total_data = (total_data - total_mean)/total_std

bn_data1 = total_data[0:100, :]
bn_data2 = total_data[100:300, :]
bn_data3 = total_data[300:360, :]
bn_data4 = total_data[360:480, :]

x_bn_data1 = bn_data1[:,[0]]
y_bn_data1 = bn_data1[:,[1]]
plt.scatter(x_bn_data1, y_bn_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度

x_bn_data2 = bn_data2[:,[0]]
y_bn_data2 = bn_data2[:,[1]]
plt.scatter(x_bn_data2, y_bn_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

x_bn_data3 = bn_data3[:,[0]]
y_bn_data3 = bn_data3[:,[1]]
plt.scatter(x_bn_data3, y_bn_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度

x_bn_data4 = bn_data4[:,[0]]
y_bn_data4 = bn_data4[:,[1]]
plt.scatter(x_bn_data4, y_bn_data4, color=sns.color_palette()[3], marker='.',alpha=0.9) # alpha设置透明度

plt.scatter(0, 0, color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.xlim((-6, 6))
plt.ylim((-6, 6))


# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'bn_data.png')
plt.savefig(fig_path)
plt.figure()
plt.show()


#### l-infinite scaling
x_lp_data1 = x_data1/np.max(np.abs(x_data1))
y_lp_data1 = y_data1/np.max(np.abs(y_data1))
plt.scatter(x_lp_data1, y_lp_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度

x_lp_data2 = x_data2/np.max(np.abs(x_data2))
y_lp_data2 = y_data2/np.max(np.abs(y_data2))
plt.scatter(x_lp_data2, y_lp_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

x_lp_data3 = x_data3/np.max(np.abs(x_data3))
y_lp_data3 = y_data3/np.max(np.abs(y_data3))
plt.scatter(x_lp_data3, y_lp_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度

x_lp_data4 = x_data4/np.max(np.abs(x_data4))
y_lp_data4 = y_data4/np.max(np.abs(y_data4))
plt.scatter(x_lp_data4, y_lp_data4, color=sns.color_palette()[4], marker='.',alpha=0.9) # alpha设置透明度


total_lp_data_x = np.concatenate((x_lp_data1,x_lp_data2,x_lp_data3,x_lp_data4), axis=0)
total_lp_data_y = np.concatenate((y_lp_data1,y_lp_data2,y_lp_data3,y_lp_data4), axis=0)

plt.scatter(np.mean(total_lp_data_x), np.mean(total_lp_data_y), color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度
plt.xlim((-1, 1))
plt.ylim((-1, 1))


# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'lp_data.png')
plt.savefig(fig_path)
plt.figure()
plt.show()



### l-infinite scaling+bn
total_mean_lp_data_x = np.mean(total_lp_data_x)
total_mean_lp_data_y = np.mean(total_lp_data_y)
total_std_lp_data_x = np.std(total_lp_data_x)
total_std_lp_data_y = np.std(total_lp_data_y)
total_lp_bn_data_x = (total_lp_data_x-total_mean_lp_data_x)/total_std_lp_data_x
total_lp_bn_data_y = (total_lp_data_y-total_mean_lp_data_y)/total_std_lp_data_y

x_lp_bn_data1 = total_lp_bn_data_x[0:100]
x_lp_bn_data2 = total_lp_bn_data_x[100:300]
x_lp_bn_data3 = total_lp_bn_data_x[300:360]
x_lp_bn_data4 = total_lp_bn_data_x[360:480]

y_lp_bn_data1 = total_lp_bn_data_y[0:100]
y_lp_bn_data2 = total_lp_bn_data_y[100:300]
y_lp_bn_data3 = total_lp_bn_data_y[300:360]
y_lp_bn_data4 = total_lp_bn_data_y[360:480]


plt.scatter(x_lp_bn_data1, y_lp_bn_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度
plt.scatter(x_lp_bn_data2, y_lp_bn_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度
plt.scatter(x_lp_bn_data3, y_lp_bn_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度
plt.scatter(x_lp_bn_data4, y_lp_bn_data4, color=sns.color_palette()[3], marker='.',alpha=0.9) # alpha设置透明度

plt.scatter(0, 0, color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'lp_bn_data.png')
plt.savefig(fig_path)
plt.figure()
plt.show()


#### graph norm
x_gn_data1 = (x_data1 - np.mean(x_data1))/np.std(x_data1)
y_gn_data1 = (y_data1 - np.mean(y_data1))/np.std(y_data1)
plt.scatter(x_gn_data1, y_gn_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度

x_gn_data2 = (x_data2 - np.mean(x_data2))/np.std(x_data2)
y_gn_data2 = (y_data2 - np.mean(y_data2))/np.std(y_data2)
plt.scatter(x_gn_data2, y_gn_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

x_gn_data3 = (x_data3 - np.mean(x_data3))/np.std(x_data3)
y_gn_data3 = (y_data3 - np.mean(y_data3))/np.std(y_data3)
plt.scatter(x_gn_data3, y_gn_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度

x_gn_data4 = (x_data4 - np.mean(x_data4))/np.std(x_data4)
y_gn_data4 = (y_data4 - np.mean(y_data4))/np.std(y_data4)
plt.scatter(x_gn_data4, y_gn_data4, color=sns.color_palette()[3], marker='.',alpha=0.9) # alpha设置透明度

total_gn_x_data = np.concatenate((x_gn_data1, x_gn_data2, x_gn_data3, x_gn_data4), axis=0)
total_gn_y_data = np.concatenate((y_gn_data1, y_gn_data2, y_gn_data3, y_gn_data4), axis=0)
total_gn_x_mean = np.mean(total_gn_x_data, axis=0)
total_gn_y_mean = np.mean(total_gn_y_data, axis=0)
plt.scatter(total_gn_x_mean, total_gn_y_mean, color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.scatter(0, 0, color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'gn_data.png')
plt.savefig(fig_path)
plt.figure()
plt.show()


#### mn 
total_data = (total_data - total_mean)/total_std

total_mn_x_data = (total_gn_x_data-total_gn_x_mean)/np.std(total_gn_x_data)
total_mn_y_data = (total_gn_y_data-total_gn_y_mean)/np.std(total_gn_y_data)

x_mn_data1 = total_mn_x_data[0:100]
y_mn_data1 = total_mn_y_data[0:100]
plt.scatter(x_mn_data1, y_mn_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度

x_mn_data2 = total_mn_x_data[100:300]
y_mn_data2 = total_mn_y_data[100:300]
plt.scatter(x_mn_data2, y_mn_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

x_mn_data3 = total_mn_x_data[300:360]
y_mn_data3 = total_mn_y_data[300:360]
plt.scatter(x_mn_data3, y_mn_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度

x_mn_data4 = total_mn_x_data[360:480]
y_mn_data4 = total_mn_y_data[360:480]
plt.scatter(x_mn_data4, y_mn_data4, color=sns.color_palette()[3], marker='.',alpha=0.9) # alpha设置透明度

plt.scatter(0, 0, color=sns.color_palette()[5], marker='o',alpha=0.9) # alpha设置透明度

plt.xlim((-3, 3))
plt.ylim((-3, 3))


# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'mn_data.png')
plt.savefig(fig_path)
plt.figure()
plt.show()