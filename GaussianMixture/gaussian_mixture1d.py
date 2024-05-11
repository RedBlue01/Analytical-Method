from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv')
data=data['电流'].values
data = data.reshape(-1, 1)  # 将数据转换为列向量

gmm = GaussianMixture(n_components=3)  # 设置聚类数目
gmm.fit(data)

labels = gmm.predict(data)

import matplotlib.pyplot as plt

plt.scatter(data, np.zeros_like(data), c=labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.savefig('/home/visitor/Huang/Analytical-Method/GaussianMixture/1d.png')
