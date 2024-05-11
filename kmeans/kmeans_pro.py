import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.font_manager as fm

# 从CSV文件读取数据
data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv')

# 将第3列提取为特征数据（X）
X = data.iloc[:, [0, 2]].values

#训练模型
k=2
kmeans = KMeans(n_clusters=k, n_init=10)
minimeans=MiniBatchKMeans(n_clusters=k, n_init=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
y_kmeans2 = minimeans.fit_predict(X)

font_path = "/home/visitor/Huang/Analytical-Method/simfang.ttf"  # 字体文件路径
font_prop = fm.FontProperties(fname=font_path)

# 绘制KMeans聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('KMeans 聚类结果', fontproperties=font_prop)
plt.savefig('/home/visitor/Huang/Analytical-Method/kmeans/kmeans.png')  # 保存图像

# 绘制MiniBatchKMeans聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans2, s=50, cmap='viridis')
centers2 = minimeans.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)
plt.title('MiniBatchKMeans 聚类结果', fontproperties=font_prop)
plt.savefig('/home/visitor/Huang/Analytical-Method/kmeans/minibatch_kmeans.png')  # 保存图像
