import pandas as pd

# 读取csv文件
df = pd.read_csv('/home/visitor/Huang/Analytical-Method/df1_processed.csv')

# 保留指定的行数
specified_rows = 1000000  # 修改为你想要保留的行数
df = df.head(specified_rows)

# 选择指定的两列
df = df.iloc[:, [0, 1]]  # 索引从0开始

# 删除包含NaN值的行
# df = df.dropna()

# 保存到新的csv文件
df.to_csv('/home/visitor/Huang/Analytical-Method/df1_processed_电流.csv', index=False)

print('Done')