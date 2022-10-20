#import python modules
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
df_train=pd.read_csv("/Users/ningduan/Python/MachineLearning/PCA_Unsuper/heart_train-1.csv")
print(df_train)
#create the data array for the input features (i.e., variables)
X_train=df_train[df_train.columns[:-1]].to_numpy()
y_train=df_train[df_train.columns[-1]].to_numpy()
#z-score the data array
X_train_mean=X_train.mean(axis=0)
X_train_std=X_train.std(axis=0)
X_train_z=(X_train-X_train_mean)/X_train_std
#create a principal component analysis (PCA) model that uses two components
pca_2=PCA(n_components=2)
#Train the two compoent PCA model on the data array
Y_2=pca_2.fit_transform(X_train)
#create a 2d scatter plot for the two PCA components
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.8])
ax.scatter(Y_2[:,0],Y_2[:,1])
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
plt.title('2D PCA Plot')
#plt.show()

#create a principle component analysis (PCA) model that uses three components
pca_3=PCA(n_components=3)
#train the three component PCA model on the data array
Y_3=pca_3.fit_transform(X_train_z)

#create a 3d scatter plot for the three PCA components
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(Y_3[:,0],Y_3[:,1],Y_3[:,2])
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('x_3')
plt.title('3D PCA plot')
#plt.show()

#create a pca model that using all components
pca=PCA()
Y_train=pca.fit_transform(X_train_z)

pc_values=np.arange(pca.n_components_) + 1
plt.plot(pc_values,pca.explained_variance_ratio_,'ro-',linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Component')
plt.xticks(pc_values)
plt.ylabel('Percent Variance Explained')
#plt.show()

pc_values=np.arange(pca.n_components_) + 1
plt.plot(pc_values,np.cumsum(pca.explained_variance_ratio_),'ro-',linewidth=2)
plt.xlabel('Component')
plt.xticks(pc_values)
plt.ylabel('Percent Variance Explained')
#plt.show()

#create a validation data split
df_validation =pd.read_csv('/Users/ningduan/Python/MachineLearning/PCA_Unsuper/heart_validation.csv')
#create validation data arrays
X_validation=df_validation[df_validation.columns[:-1]].to_numpy()
y_validation=df_validation[df_validation.columns[-1]].to_numpy()
#z-score the validation input data array
X_validation_z=(X_validation-X_train_mean)/X_train_std
#grid search over number of pca components so select the number of components
#based on validation set and logistic regression
models={}
pcas={}
train_accs=[]
validation_accs=[]
for i, j in enumerate(range(13)):
  k=j+1
  print('k=' + str(k))
  logistic=LogisticRegression(max_iter=10000,penalty='none')
  pca=PCA(n_components=k)
  pca.fit(X_train_z)
  pcas[k]=pca
  X_train_pca=pca.transform(X_train_z)
  logistic.fit(X_train_pca,y_train)
  models[k]=logistic
  y_train_preds=logistic.predict(X_train_pca)
  print('Train Accuracy')
  train_acc=np.equal(y_train_preds,y_train).mean()
  print(train_acc)
  train_accs.append(train_acc)
  print('Validation Accuracy')
  X_validation_pca=pca.transform(X_validation_z)
  validation_acc=np.equal(logistic.predict(X_validation_pca),y_validation).mean()
  print(validation_acc)
  validation_accs.append(validation_acc)

#load the heart disease test data csv and create the test data arrays
df_test=pd.read_csv('/Users/ningduan/Python/MachineLearning/PCA_Unsuper/heart_test-1.csv')
X_test=df_test[df_test.columns[:-1]].to_numpy()
y_test=df_test[df_test.columns[-1]].to_numpy()

#z-score the test data array
X_test_z=(X_test-X_train_mean)/X_train_std

#compare the model performance on the validation and test data for various numbers of pca components
for i, j in enumerate(range(13)):
  k=j+1
  print()
  print('k=' + str(k))
  print('Train Accuracy',train_accs[i])
  print('Validation Accuracy',validation_accs[i])
  print(np.equal(models[k].predict(pcas[k].transform(X_test_z)), y_test).mean())
#use logistic regression check the accuracy of pca
