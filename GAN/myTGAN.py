# import pandas as pd
# data=pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv')
# continuous_columns=[0,2]
# 原来我在Conda上安装了Python3.8(文档说它只适用于Python3.5、3.6和3.7)，在降级到Python3.7.5之后，这个包可以工作。
from tgan.data import load_demo_data
data, continuous_columns = load_demo_data('census')
data.head(3).T[:10]
print('Done')