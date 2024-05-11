from scipy.cluster.hierarchy import linkage     #导入linage函数用于层次聚类
from scipy.cluster.hierarchy import dendrogram  #dendrogram函数用于将聚类结果绘制成树状图
from scipy.cluster.hierarchy import fcluster    #fcluster函数用于提取出聚类的结果
from sklearn.datasets import make_blobs         #make_blobs用于生成聚类算法的测试数据
from sklearn.cluster import AgglomerativeClustering  #自底向上层次聚类算法
import matplotlib.pyplot as plt                 #导入matplotlib绘图工具包

#生成测试数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.4, random_state=0)
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c='b')
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/hierarchical_before.png')
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
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/hierarchical_tree.png')

#根据临界距离返回聚类结果
d = 15
labels_1 = fcluster(Z, t=d, criterion='distance')
print(labels_1[: 100])  # 打印聚类结果
print(len(set(labels_1)))  # 看看在该临界距离下有几个 cluster

#根据聚类数目返回聚类结果
k = 3
labels_2 = fcluster(Z, t=k, criterion='maxclust')
print(labels_2[: 100])
list(labels_1) == list(labels_2)  # 看看两种不同维度下得到的聚类结果是否一致

#聚类的结果可视化，相同的类的样本点用同一种颜色表示
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=labels_2, cmap='prism')
plt.savefig('/home/visitor/Huang/Analytical-Method/hierarchical_clustering/hierarchical_after.png')
