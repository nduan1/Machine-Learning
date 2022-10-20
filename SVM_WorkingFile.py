import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

#classify the data to certain group, we want to find the largest distance value of closest point to the hyperplain
#we want to turn this data into a form where we can draw a hyperplain through a divided training data
#kernel is a function (e.g. we want to divide two dimension(x1,x2) data points to two group. But sometime it is hard to group it
#you can use kernel create a third (x3) to help divide the data points at 3D, if it is still head to group, we can add another dimension)
#support vector, soft margin
#SVM to deal with high dimension, KNN works the similar, but not well for huge dimensions
cancer=datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

X=cancer.data
y=cancer.target

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2)

#print(x_train,y_train)
classes=['malignant', 'benign']

#clf=svm.SVC(kernel="poly", degree=2)
clf=svm.SVC(kernel="linear", C=2)#C=0 means hard margin,C define the soft margin, C=2 means double the points allow the
clf.fit(x_train, y_train)

y_pred=clf.predict(x_test)

acc=metrics.accuracy_score(y_test,y_pred)

print(acc)



