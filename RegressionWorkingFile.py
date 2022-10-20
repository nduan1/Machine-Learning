import pandas as pd
import tensorflow
import keras
import sklearn
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style

import pickle

data=pd.read_csv("student-mat.csv", sep=";")

print(data.head()) #rows are observations

data=data[["G1","G2","G3","studytime","failures","absences"]]

print(data.head())

predict="G3" #label is G3, we want to predict

X=np.array(data.drop(["G3"],1))#attribute
y=np.array(data[predict])#response variable
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1) # we also want test data after getting the best model

""" 
best=0
for _ in range(30): #we don't care about this variable so use underscore
    # spliting the data 10% to test data, others are training data
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

    #training process and generate and save the model
    linear=linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)#give the accuracy of the model
    print(acc)
    if acc > best:
        best=acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f) #dump model into the file f, this will save the model

"""
# this for loop is used to get a better model with high accuracy

pickle_in=open("studentmodel.pickle", "rb")#open the saved model and use it
linear=pickle.load(pickle_in)#load model into the variable called linear

print("Co: \n", linear.coef_)
print("intercept: \n",linear.intercept_)

predictions=linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])# print out the predictions and actual y

###scatter plot ###
p="studytime"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()


