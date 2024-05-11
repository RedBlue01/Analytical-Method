import pandas as pd

# 读取原始 CSV 文件
original_data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv')

# 对原始数据进行随机抽样，抽样10000行
sampled_data = original_data.sample(n=10000)

# 将抽样后的数据保存到新的 CSV 文件中
sampled_data.to_csv('/home/visitor/Huang/Analytical-Method/column_123mini.csv', index=False)
