#svc(support vector classifier)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import pandas as pd


def plot_decision_boundaries(X, y, model):
    # adapted from https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Utilities/ML-Python-utils.py

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor="k")
    plt.show()


def plot3D(X, y, mod=None):
    # code adapted from https://stackoverflow.com/questions/51278752/visualize-2d-3d-decision-surface-in-svm-scikit-learn

    z = lambda x, y: (-mod.intercept_[0] - mod.coef_[0][0] * x - mod.coef_[0][1] * y) / mod.coef_[0][2]

    tmp = np.linspace(-1, 1, 30)
    xx, yy = np.meshgrid(tmp, tmp)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X['x1'][y == 0], X['x2'][y == 0], X['sq'][y == 0], 'o', c='purple')
    ax.plot3D(X['x1'][y == 1], X['x2'][y == 1], X['sq'][y == 1], 'oy')
    if mod:
        ax.plot_surface(xx, yy, z(xx, yy), alpha=0.5)
    ax.view_init(30, 60)
    plt.show()


np.random.seed(0)  # ensure consistent results

#SVM trying to find a separating line (or, more generally, a hyperplane) between two classes.
#First, however, we'll look at some data that cannot be separated by a line. We'll perform feature engineering to prepare it for SVM.

#We'll start with a synthetic dataset with concentric circles, stored in a pandas DataFrame, with the two features called x1 and x2.

from sklearn.datasets import make_circles
import pandas as pd
X,y=make_circles(n_samples=1000,factor=0.3, noise=0.05)#create a sample dataset with two circle
X=pd.DataFrame(X)
X.columns=["x1","x2"]
#print(X)
#print(y)#y is the label
plt.figure(figsize=(6,6))
plt.scatter(X["x1"],X["x2"],c=y)
plt.show()

#we can't use regular line to cluster the data, so we "engineer" or create a line to seperate it.
#this will have the effect of projecting it into higher dimensions.
X['sq']=X['x1']**2+X['x2']**2
#print(X)
#plot3D(X,y)
#create a svc model with a linear kernel named mod to fit it to X and y
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
mod=SVC(kernel="linear")
mod.fit(X,y)
#plot3D(X,y,mod)

####by engineering features we could use standard SVM. But what if the relationship is not as obvious
#Kernel can help. Using a polynomial kernel of degree 2 will have the effect of adding such features for all quadratic
#combinations of the original features, but without actually computing new features.
X=X.drop(['sq'], axis=1)
X=X.to_numpy()
#print(X)
#make mod an svc model with a polynomial kernel of degree 2, fit it to X and y
mod=make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2))
mod.fit(X,y)
plot_decision_boundaries(X,y,mod)
#what degree of polynomial do we need?we can look at the Radial Basis Function(RBF) also
#called a gaussian kernel. This generlizes the idea of adding features up to infinite dimensions(remeber
# they are not actually computed). Since the RBF is based on the distance of training points to each other,
#this is also makes the resuting boundary similar to KNN.
##the gamma hyperparameter controls how much training points influence each other as they get further apart
#we will go back to our breast cancer dataset, get two features, scale and split into train and test sets

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

(data,labels)=load_breast_cancer(return_X_y=True,as_frame=True)
var1="mean radius"
var2='mean compactness'
X=data[[var1,var2]].to_numpy()
y=labels
ss=StandardScaler()
X=ss.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#make an SVC model named mod with an RBF kernel and gamma of 50, and fit it to x_train and y_tarin
mod=make_pipeline(StandardScaler(),SVC(kernel='rbf',gamma=1))# you can try different gamma, the system will use a
# formula to set it to a value that usually works well.
mod.fit(X_train,y_train)
#let's see the decision boundary and evaluate
from sklearn.metrics import  accuracy_score
plot_decision_boundaries(X_train,y_train,mod)
print("Training acc: %f"% accuracy_score(y_train,mod.predict(X_train)))
print("Training acc: %f"% accuracy_score(y_test,mod.predict(X_test))) #these two accuracy are much closer if the gamma is 1 in this case
