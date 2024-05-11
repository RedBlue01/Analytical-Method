import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

# 从CSV文件中读取数据
data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv')

# 随机采样
sample_size = 10000  # 样本数量
random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
sampled_data = data.iloc[random_indices]

# 提取时间戳和电流大小列
timestamps = sampled_data['Time'].values
currents = sampled_data['电流'].values
X = np.column_stack((timestamps, currents))
print(f'随机采样后的数据量：{len(X)}')

# 创建一个高斯混合模型对象
gmm = GaussianMixture(n_components=2)

# 将数据拟合到模型中
gmm.fit(X)

# 聚类标签
labels = gmm.predict(X)

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.savefig('/home/visitor/Huang/Analytical-Method/GaussianMixture/gaussian.png')
