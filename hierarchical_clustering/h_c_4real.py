from scipy.cluster.hierarchy import linkage     #导入linage函数用于层次聚类
from scipy.cluster.hierarchy import dendrogram  #dendrogram函数用于将聚类结果绘制成树状图
from scipy.cluster.hierarchy import fcluster    #fcluster函数用于提取出聚类的结果
from sklearn.cluster import AgglomerativeClustering  #自底向上层次聚类算法
import matplotlib.pyplot as plt                 #导入matplotlib绘图工具包
import pandas as pd
import numpy as np

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

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c='b')
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/1hierarchical_before.png')

#from scipy.cluster.hierarchy import linkage
#层次聚类实现
#from scipy.cluster.hierarchy import dendrogram
Z = linkage(X,  method='ward', metric='euclidean')
print(Z.shape)
print(Z[: 5])

#画出树状图
#from scipy.cluster.hierarchy import fcluster
plt.figure(figsize=(10, 8))
dendrogram(Z, truncate_mode='lastp', p=20, show_leaf_counts=False, leaf_rotation=90, leaf_font_size=15,
           show_contracted=True)
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/1hierarchical_tree.png')

#根据临界距离返回聚类结果
d = 15
labels_1 = fcluster(Z, t=d, criterion='distance')
# print(labels_1[: 100])  # 打印聚类结果
print(len(set(labels_1)))  # 看看在该临界距离下有几个 cluster

#根据聚类数目返回聚类结果
k = 3
labels_2 = fcluster(Z, t=k, criterion='maxclust')
# print(labels_2[: 100])
print(list(labels_1) == list(labels_2))  # 看看两种不同维度下得到的聚类结果是否一致

#聚类的结果可视化，相同的类的样本点用同一种颜色表示
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels_2, cmap='prism')
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/1hierarchical_after.png')
