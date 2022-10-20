#import python modules
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load csv with patient demographic information
df_train=pd.read_csv('/Users/ningduan/Python/MachineLearning/KMeans_Unsuper/heart_train-1.csv')
#create input training data array
X_train=df_train[df_train.columns[:-1]].to_numpy()
#z-score the data array
X_train_mean=X_train.mean(axis=0)
X_train_std=X_train.std(axis=0)
X_train_z=(X_train-X_train_mean)/X_train_std

#apply k-means using two clusters to the z-scored input data array
kMeans_2=KMeans(n_clusters=2)

#apply pca to the input data array using two components
pca_2=PCA(n_components=2)
X_train_2_pca=pca_2.fit_transform(X_train_z)

#get the centroids for k-means using k=2 in two-component pca space
means=kMeans_2.fit(X_train_2_pca).cluster_centers_
means

#plot the pca transformed input data and the corresponding centroids for k-means with k=2
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.9])
ax.scatter(X_train_2_pca[:,0],X_train_2_pca[:,1],c='k')
ax.scatter(means[0,0],means[0,1],c='r')
ax.scatter(means[1,0],means[1,1],c='b')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
plt.title('2D PCA Plot with k-means')

#calculate the intertias for different values of k, inertia measures how well a dataset was clustered by k-means
inertias=[]
ks=[]
for i in range(10):
  k=i+1
  ks.append(k)
  print('K='+str(k))
  kmeans=KMeans(n_clusters=k)
  kmeans.fit(X_train_2_pca)
  print('inertia='+str(kmeans.inertia_))
  inertias.append(kmeans.inertia_)

#create an inertia elbow plot
xs=np.array(list(range(len(ks))))
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.9])
ax.plot(xs,inertias)
ax.set_xticks(xs)
ax.set_xticklabels(ks)
ax.set_xlabel('k')
ax.set_ylabel('Inertia')




#apply pca to the input data array using four components
kMeans_4=KMeans(n_clusters=4)
kMeans_4.fit(X_train_2_pca)

#plot the pca transformed input data and the corresponding centroids for k-means with k=4
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.9])
ax.scatter(X_train_2_pca[:,0],X_train_2_pca[:,1],c='k')
colors=['r','b','g','y']
for i in range(4):
  ax.scatter(kMeans_4.cluster_centers_[i,0],kMeans_4.cluster_centers_[i,1],c=colors[i])
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
plt.title('2D PCA Plot with k-means')
plt.show()