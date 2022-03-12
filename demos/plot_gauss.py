import os
import math
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(__file__)


u = 0   # 均值μ
sig = math.sqrt(1)  # 标准差δ
x = np.linspace(u - 3*sig, u + 3*sig, 20000)   # 定义域
y1 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
sig = np.std(y1)
y1 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
plt.plot(x, y1, "g", linewidth=2)    # 加载曲线


sig = math.sqrt(2)  # 标准差δ
y2 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
sig = np.std(y2)
y2 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
plt.plot(x, y2, "r", linewidth=2)    # 加载曲线


y_mix = np.concatenate([y1,y2])
sig = np.std(y_mix)
y_mix = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
plt.plot(x, y_mix, "b", linewidth=2)    # 加载曲线

plt.savefig(os.path.join(dir_path, f'gaus1.png'))
plt.close()  # 显示
