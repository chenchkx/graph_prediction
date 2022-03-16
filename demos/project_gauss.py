

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

plt.xlim((-6, 6))
plt.ylim((-6, 6))

# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'gauss_distribution.png')
plt.savefig(fig_path)
plt.figure()
plt.show()



project_matrix = np.random.random((2,2))
project_bais = np.random.random(2)
project_data1 = data1.dot(project_matrix) + project_bais
project_x_data1 = project_data1[:,[0]]
project_y_data1 = project_data1[:,[1]]
plt.scatter(project_x_data1, project_y_data1, color=sns.color_palette()[0], marker='.',alpha=0.9) # alpha设置透明度


project_data2 = data2.dot(project_matrix) + project_bais
project_x_data2 = project_data2[:,[0]]
project_y_data2 = project_data2[:,[1]]
plt.scatter(project_x_data2, project_y_data2, color=sns.color_palette()[1], marker='.',alpha=0.9) # alpha设置透明度

project_data3 = data3.dot(project_matrix) + project_bais
project_x_data3 = project_data3[:,[0]]
project_y_data3 = project_data3[:,[1]]
plt.scatter(project_x_data3, project_y_data3, color=sns.color_palette()[2], marker='.',alpha=0.9) # alpha设置透明度



plt.xlim((-6, 6))
plt.ylim((-6, 6))

# plt.axis('off')
# plt.legend(fontsize=14, loc='upper left')
plt.tick_params(labelsize=14) # 设置坐标刻度尺大小
fig_path = os.path.join(dir_path, 'gauss_project.png')
plt.savefig(fig_path)
plt.figure()
plt.show()