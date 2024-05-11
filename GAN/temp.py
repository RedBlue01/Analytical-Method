import pandas as pd

df=pd.read_csv('/home/visitor/Huang/Analytical-Method/GAN/tgan_sampleDATA.csv')

# 删除第一列
df = df.drop(df.columns[0], axis=1)

df.to_csv('/home/visitor/Huang/Analytical-Method/GAN/new_file.csv', index=False)  # 写入新文件