import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123.csv')

# 去除第三列为空值的数据
df = df.dropna(subset=[df.columns[2]])

# 选择要保存的列
selected_columns = df.iloc[:, :3]  # 选择第1、2和3列

# 写入到新的 CSV 文件
selected_columns.to_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv', index=False)
