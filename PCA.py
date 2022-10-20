import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import statistics as stat
from sklearn.decomposition import PCA

#import data
(data, labels)=load_breast_cancer(return_X_y=True,as_frame=True)
print(data)
print(labels)
#use standardscaler to normalize the data, using the fit_transform() function(remember to import the module).
#Store the scale data in scl.

scaler=preprocessing.StandardScaler()
scaler.fit(data)
scl=scaler.fit_transform(data)
#print(scaler)
#print(scl)
print("Col 0 mean: %f" % stat.mean(scl[:,0]))
print("Col 0 stddev: %f" % stat.stdev(scl[:,0]))

#fit pca model to a scaled data that keeps only 2 components that explain the most variance.
pca=PCA(n_components=2)
pca.fit(scl)
print(pca.explained_variance_ratio_)# the 2 component explained a lot of variance, which means our variable
#are highly correlated and thus very redundant.

#transform the scaled data based on our PCA model and store it in tfm
tfm=pca.transform(scl)
print(tfm.shape)

plt.scatter(tfm[:,0],tfm[:,1],c=labels)
plt.show()


