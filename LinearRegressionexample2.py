from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


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


np.random.seed(0)  # ensure consistent results

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True)
print(X.corr())#We can see that some of the serum measurements (e.g. "s2" and "s3") are highly correlated.
# fit a LR model
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
lr = LinearRegression().fit(X_train, y_train)

print("Coefficients:")
print(list(zip(X.columns, lr.coef_)))
print()
print("Train R^2: %f"%lr.score(X_train, y_train))
print("Test R^2:  %f"%lr.score(X_test, y_test))

#LASSO(L1 panelty)
#Fit a Lasso model named lr to the training data.
#Lasso stands for least absolute shrinkage and selection operator.
# Pay attention to the words, “least absolute shrinkage” and “selection”.
# Lasso regression is used in machine learning to prevent overfitting.
# It is also used to select features by setting coefficients to zero
#Optimizing the LASSO loss function does result in some of the weights becoming zero.
# Thus, some of the features will be removed as a result.
# This is why LASSO regression is considered to be useful as a supervised feature selection technique.
#Use LassoCV implementation for applying cross-validation to Lasso regression.
from sklearn import linear_model
lr = linear_model.Lasso().fit(X_train, y_train)
print("Coefficients:")
print(list(zip(X.columns, lr.coef_)))
print()
print("Train R^2: %f"%lr.score(X_train, y_train))
print("Test R^2:  %f"%lr.score(X_test, y_test))# we want to remove the variables with coeff is zero

#___________________lasso and ridge (l1 and l2 regularization)_____________#
#LASSO regression (L1 panelty) adds"absolute value of mangnitude" of coefficient as penalty term to the loss function

#ridge regression (L2 panelty) adds"squared mangnitude" of coefficient as penalty term to the loss function
#for l1 and l2, if lamda is too large, and add too much weight on loss function, would result in underfitting
# the key differences between these is that lasso shrinks the less important features's coefficient to zero
#l2 regularization doesn't perform feature selection, since weights are only reduced to values near 0 instead of 0.
#l1 regularization has built in feature selection
#l1 regularization is robust to outliers, l2 regularization is not
#l1 is more robust and has more possiblities, more compuationally expensive, and create a sparse output

#Logistic regression
(data, labels) = load_breast_cancer(return_X_y=True, as_frame=True)
var1 = 'mean radius'
var2 = 'mean compactness'
X = data[[var1, var2]].to_numpy()
y = labels.to_numpy()
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Fit a LogisticRegression model called lr to X_train and y_train.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
plot_decision_boundaries(X_train, y_train, lr)
print("Training acc: %f"%accuracy_score(y_train, lr.predict(X_train)))
print("Testing acc:  %f"%accuracy_score(y_test, lr.predict(X_test)))

#In Scikit-learn, classifiers that use cut-offs (like LogisticRegression)
# can provide raw prediction probabilities instead of labels using predict_proba.
# This will return an array with a column for each label (even if there are only two).

#6. Use the predict_proba function of lr to generate raw prediction probabilities for X_test.
# Use slicing to extract the second column (for label "1") and store in proba.
proba=lr.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve,RocCurveDisplay,roc_auc_score

fpr, tpr, _ = roc_curve(y_test, proba)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

print("AUC: %f"%roc_auc_score(y_test, proba))