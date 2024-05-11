import pandas as pd

data = pd.read_csv('/home/visitor/Huang/Analytical-Method/df1_processed_电流.csv')

savepath='/home/visitor/Huang/Analytical-Method/GAN/'
savename='CTGANsynthesizer_e100'
# 控制保存的元数据与模型的名字

# 创建元数据
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.save_to_json(filepath=savepath+savename+'.json')

python_dict = metadata.to_dict()
# print(data)
print(python_dict)

print(data)

# CTGAN Synthesizer 使用基于 GAN 的深度学习方法来训练模型并生成合成数据
from sdv.single_table import CTGANSynthesizer
synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=True,
    epochs=100,
    verbose=True,
    cuda=True
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
synthesizer.get_loss_values().to_csv(savepath+savename+'_loss.csv')

import torch
torch.save(synthesizer,savepath+savename+'torch.pkl')

synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv(savepath+savename+'_sample.csv', index=False)

print(synthetic_data)

synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.to_csv(savepath+savename+'_sample2.csv', index=False)

print(synthetic_data)
print('Done')