import pandas as pd

# 使用 pd.read_csv() 函数读取 CSV 文件，仅读取第 0 列和第 2 列
# 指定 usecols 参数为一个整数列表，列表中包含要读取的列的索引（从0开始计数）
data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123after.csv', usecols=[0, 2])

# 创建元数据
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata()
metadata.detect_from_dataframe(data)
python_dict = metadata.to_dict()
print(data)
print(python_dict)

# CTGAN Synthesizer 使用基于 GAN 的深度学习方法来训练模型并生成合成数据
from sdv.single_table import CTGANSynthesizer
synthesizer=CTGANSynthesizer.load(
    filepath='/home/visitor/Huang/Analytical-Method/GAN/my_synthesizer_60.pkl'
)
synthesizer.save(
    filepath='/home/visitor/Huang/Analytical-Method/GAN/my_synthesizer_80.pkl'
)
synthesizer.epochs=0
print(synthesizer.get_parameters())
# 使用参数定义约束，然后将其添加到合成器中
""" my_constraint={
    'constraint_class':'Positive',
    'constraint_parameters':{
        'column_name':'电流',
        'strict_boundaries':False
        # /home/visitor/anaconda3/envs/AM/lib/python3.10/site-packages/sdv/constraints/tabular.py中strict参数已更改为strict_boundaries
    }
}
synthesizer.add_constraints(constraints=[
    my_constraint
]) """

synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=10)

print(synthetic_data)
print('Done')