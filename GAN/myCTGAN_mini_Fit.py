import pandas as pd

# 使用 pd.read_csv() 函数读取 CSV 文件，仅读取第 0 列和第 2 列
# 指定 usecols 参数为一个整数列表，列表中包含要读取的列的索引（从0开始计数）
data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123mini.csv', usecols=[0, 2])

savepath='/home/visitor/Huang/Analytical-Method/GAN/'
savename='my_synthesizer_mini_e100NEW'
# 控制保存的元数据与模型的名字

# 创建元数据
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.save_to_json(filepath=savepath+savename+'.json')

python_dict = metadata.to_dict()
# print(data)
print(python_dict)

# CTGAN Synthesizer 使用基于 GAN 的深度学习方法来训练模型并生成合成数据
from sdv.single_table import CTGANSynthesizer
synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=True,
    epochs=1000,
    verbose=True
)
synthesizer.save(
    filepath=savepath+savename+'.pkl'
)
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
synthesizer.get_loss_values()

import torch
torch.save(synthesizer,savepath+'my_awesome_model.pkl')

synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv('/home/visitor/Huang/Analytical-Method/GAN/CTGAN_sample.csv', index=False)

print(synthetic_data)

synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv('/home/visitor/Huang/Analytical-Method/GAN/CTGAN_sample2.csv', index=False)

print(synthetic_data)
print('Done')