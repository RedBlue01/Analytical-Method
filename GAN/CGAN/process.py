import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cGAN import cGAN
import numpy as np

df=pd.read_csv('/home/visitor/Huang/Analytical-Method/GAN/column_123mini.csv')

scaler = StandardScaler()

X = scaler.fit_transform(df.drop('电流', 1))
y = df['电流'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training cGAN
cgan = cGAN()
y_train = y_train.reshape(-1,1)
pos_index = np.where(y_train==1)[0]
neg_index = np.where(y_train==0)[0]
cgan.train(X_train, y_train, pos_index, neg_index, epochs=500)