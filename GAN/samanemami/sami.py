from Gan_definition import Gan
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('/home/visitor/Huang/Analytical-Method/GAN/column_123mini.csv')
def _df(data):
    df = pd.DataFrame(data)
    for c in range(df.shape[1]):
        mapping = {df.columns[c]: c}
        df = df.rename(columns=mapping)
    return df

X = (data.drop(columns=["电流"])).values
y = (data["电流"]).values


X = KNNImputer().fit_transform(X)
data = _df(StandardScaler().fit_transform(np.column_stack((X, y))))

model = Gan(data)
generator = model._generator()
descriminator = model._discriminator()
gan_model = model._GAN(generator=generator, discriminator=descriminator)
trained_model = model.train(
    generator=generator, discriminator=descriminator, gan=gan_model)

noise = np.random.normal(0, 1, data.shape) 
new_data = _df(data=trained_model.predict(noise))


fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.heatmap(data.corr(), annot=True, ax=ax[0], cmap="Blues")
sns.heatmap(new_data.corr(), annot=True, ax=ax[1], cmap="Blues")
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")
fig.savefig('/home/visitor/Huang/Analytical-Method/GAN/fig1.png')

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax[0].scatter(data.iloc[:, 0], data.iloc[:, 1])
ax[1].scatter(new_data.iloc[:, 0], new_data.iloc[:, 1])
ax[0].set_title("Original Data")
ax[1].set_title("synthetic Data")
fig.savefig('/home/visitor/Huang/Analytical-Method/GAN/fig2.png')