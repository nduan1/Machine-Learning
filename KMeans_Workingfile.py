#K in K means stands for how many clusters
#select k random centroids and cluster the points, until no points change the group
#cons: the speed is low, and huge calculation, calculation intensive
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits=load_digits()
data=scale(digits.data)# scale down the data
#print(data)
y=digits.target
#k=10
k=len(np.unique(y))# 10
#print(k)
samples,features=data.shape
#print(data.shape)#1797 row or observations, 64 cols or features
# we don't need to have train data or test data
def bench_k_means(estimator,name,data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name,estimator.inertia_,
             metrics.homogeneity_score(y,estimator.labels_),
             metrics.completeness_score(y,estimator.labels_),
             metrics.v_measure_score(y,estimator.labels_),
             metrics.adjusted_rand_score(y,estimator.labels_),
             metrics.adjusted_mutual_info_score(y,estimator.labels_),
             metrics.silhouette_score(y,estimator.labels_,metric='euclidean')))# the higher the better for most of them
clf=KMeans(n_clusters=k,init='random')
bench_k_means(clf,"1",data)

