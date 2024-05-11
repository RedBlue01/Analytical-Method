import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/visitor/Huang/Analytical-Method/df1.csv')

# 设置显示选项，使输出为非科学计数法
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print(df.describe())
