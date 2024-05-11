loadpath='/home/visitor/Huang/Analytical-Method/GAN/'
loadname='my_synthesizer_mini_b100'

""" from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata.load_from_json(filepath=loadpath+loadname+'.json')
synthesizer=CTGANSynthesizer(metadata)
print(synthesizer.get_info())
print(synthesizer.get_parameters())
print(synthesizer.get_metadata())
synthesizer=CTGANSynthesizer.load(
    filepath=loadpath+loadname+'.pkl'
)
print('------------------------------------------------')
print(synthesizer.get_info())
print(synthesizer.get_parameters())
print(synthesizer.get_metadata())

synthetic_data = synthesizer.sample(num_rows=1000)
synthetic_data.to_csv('/home/visitor/Huang/Analytical-Method/GAN/synthetic_data.csv', index=False)

print(synthetic_data) """

import torch
loaded_model=torch.load(loadpath+'my_awesome_model.pkl')
print('torch保存的模型')
sample=loaded_model.sample(10)
print(sample)
print('Done')

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata.load_from_json(filepath=loadpath+loadname+'.json')
synthesizer=CTGANSynthesizer(metadata)
# print(synthesizer.get_info())
# print(synthesizer.get_parameters())
# print(synthesizer.get_metadata())
synthesizer=CTGANSynthesizer.load(
    filepath=loadpath+'my_model'+'.pkl'
)
print('------------------------------------------------')
# print(synthesizer.get_info())
# print(synthesizer.get_parameters())
# print(synthesizer.get_metadata())
print('cloudpickle保存的模型')
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data.to_csv('/home/visitor/Huang/Analytical-Method/GAN/synthetic_data.csv', index=False)

print(synthetic_data)