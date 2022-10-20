#GMM can be used for clustering, which is a probabilistic model that assumes all the data points are generated
#from a mixture of a finite number of gaussian distrubutions with unknown parameters
#GMM is an example of model that doesn't need the data scaled, because it accounts for variance
#Gaussian mixture models (GMMs) are a type of machine learning algorithm.
#They are used to classify data into different categories based on the probability distribution.
# K-Means Clustering:
# It is an algorithm, which classifies samples based on attributes/features into K number of clusters. Clustering or grouping of samples is done by minimizing the distance between sample and the centroid. i.e. Assign the centroid and optimize the centroid based on the distances from the points to it. This is called as Hard Assignment i.e. We are certain that particular points belong to particular centroid and then based on the least squares distance method, we will optimize the place of the centroid.
# Advantages of K-Means:
# 1. Running Time
#
# 2. Better for high dimensional data.
#
# 3. Easy to interpret and Implement.
#
# Disadvantages of K-Means:
#
# 1. Assumes the clusters as spherical, so does not work efficiently with complex geometrical shaped data(Mostly Non-Linear)
#
# 2. Hard Assignment might lead to mis grouping.
#
# Guassian Mixture:
#
# Instead of Hard assgning data points to a cluster, if we are uncertain about the data points where they belong or to which group, we use this method. It uses probability of a sample to determine the feasibility of it belonging to a cluster.
#
# Advantages:
#
# 1. Does not assume clusters to be of any geometry. Works well with non-linear geometric distributions as well.
#
# 2. Does not bias the cluster sizes to have specific structures as does by K-Means (Circular).
#
# Disadvantages:
#
# 1. Uses all the components it has access to, so initialization of clusters will be difficult when dimensionality of data is high.
#
# 2. Difficult to interpret.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.mixture import GaussianMixture

#generate some data
np.random.seed(0)

#generate spherical data centered on (20,20)
n_samples=50
shifted_guassian=np.random.randn(n_samples,2) + np.array([20,20])
#print(shifted_guassian)
#print()
#generate zero centered streched Gaissian data
C=np.array([[0.0,-0.7],[3.5,0.7]])
stretched_gaissian=np.dot(np.random.randn(n_samples,2),C)

#print(stretched_gaissian)
#concatenate
data=np.vstack([shifted_guassian,stretched_gaissian])
#print(data.shape)#100 obs and 2 features

#plt.scatter(data[:,0],data[:,1])
#plt.show()

gm=GaussianMixture(n_components=2)
gm.fit(data)
print(gm.means_)
#print(gm.covariances_)