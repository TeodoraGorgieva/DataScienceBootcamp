import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

df = pd.read_csv('College_Data', index_col=0)
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

g2=sns.FacetGrid(df,hue='Private', palette='coolwarm', size=6, aspect=2)
g2=g2.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()
#data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)
#plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
#plt.show()
#kmeans = KMeans(n_clusters=4)
#kmeans.fit(data[0])
#print(kmeans.cluster_centers_)