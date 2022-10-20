#decision trees iteratively split data using simple rules. the classifier they create can be easily
#interpreted. However, they also suffer from overfit. this can be dealt with by pruning or by
#using random forests, which using ensembling and subsetting. A drawback of Random forests
#is that they are no longer interpretable.
#fit a DecisionTreeClassifer model named mod to X_train and y_train
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score

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


(data,labels)=load_breast_cancer(return_X_y=True,as_frame=True)
var1="mean radius"
var2='mean compactness'
X=data[[var1,var2]].to_numpy()
y=labels
ss=StandardScaler()
X=ss.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

mod=DecisionTreeClassifier(random_state=0)
mod.fit(X_train,y_train)

plot_decision_boundaries(X_train,y_train,mod)
print("Training acc: %f"% accuracy_score(y_train,mod.predict(X_train)))#1.0
print("Training acc: %f"% accuracy_score(y_test,mod.predict(X_test)))
#it looks overfit
print()
mod=DecisionTreeClassifier(random_state=0,max_depth=3)#the depth of the tree is the maximum distance
#between the root and any leaf, the default depth is none may cause the overfit, the hyperparameter
#is the how many laryers of the tree you want
mod.fit(X_train,y_train)

plot_decision_boundaries(X_train,y_train,mod)
print("Training acc: %f"% accuracy_score(y_train,mod.predict(X_train)))
print("Training acc: %f"% accuracy_score(y_test,mod.predict(X_test)))

#let us see the tree structure
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(mod, feature_names=[var1,var2],label='none',fontsize=4)
plt.show()

####random forest###
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
mod=RandomForestClassifier(max_depth=3)
mod.fit(X_train,y_train)
plot_decision_boundaries(X_train,y_train,mod)
print("Training acc: %f"% accuracy_score(y_train,mod.predict(X_train)))
print("Training acc: %f"% accuracy_score(y_test,mod.predict(X_test)))
