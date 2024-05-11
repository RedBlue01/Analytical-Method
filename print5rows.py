import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/visitor/Huang/Analytical-Method/df1 copy.csv')

# 设置显示选项，使输出为非科学计数法
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 格式化输出
with pd.option_context('display.colheader_justify','left'):
    print(df)
