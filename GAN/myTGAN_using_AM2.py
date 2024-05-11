import pandas as pd

data = pd.read_csv('/home/visitor/Huang/Analytical-Method/column_123mini.csv', usecols=[0, 2])
continuous_columns=[0,1]

from tgan.model import TGANModel
tgan=TGANModel(
    continuous_columns,
    steps_per_epoch=300,
    output='/home/visitor/Huang/Analytical-Method/GAN/myTGANmodel'
    )
tgan.save('/home/visitor/Huang/Analytical-Method/GAN/myTGANmodel.pkl')
# force=True避免被重写
tgan.fit(data)

samples=tgan.sample(10000)
samples.to_csv('/home/visitor/Huang/Analytical-Method/GAN/tgan_sampleDATA_new.csv')