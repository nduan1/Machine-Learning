import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
df_train=pd.read_csv("/Users/ningduan/Python/MachineLearning/LogisticRegression/heart_train.csv")
df_test=pd.read_csv("/Users/ningduan/Python/MachineLearning/LogisticRegression/heart_test.csv")
#print(df_train)
#target would be y
#input dataset
X_train=df_train[df_train.columns[:-1]].to_numpy()
X_test=df_test[df_test.columns[:-1]].to_numpy()
#print(X_train)
y_train=df_train[df_train.columns[-1]].to_numpy()
y_test=df_test[df_test.columns[-1]].to_numpy()
#print(y_train)
#standscaler
ss = StandardScaler()# the same thing with z-score transformation
X_train_z = ss.fit_transform(X_train)
#X_test_ss = ss.fit_transform(X_test)
#print(X_train)
X_train_mean = X_train.mean(axis=0)#select object across rows, vertically
X_train_std = X_train.std(axis=0)
# X_train_z=(X_train-X_train_mean)/X_train_std
X_test_z=(X_test-X_train_mean)/X_train_std #we are not using scaler for test dataset, we want use regular z-score calculation
print(X_test_z)
lr=LogisticRegression(max_iter=10000,penalty="none")
lr.fit(X_train_z,y_train)

#lr.fit(X_test,y_test)
#print()
#print("Train R^2: %f"%lr.score(X_train, y_train))
#print("Test R^2:  %f"%lr.score(X_test, y_test))
#print()
#print("Coefficients:")
#print(list(zip(df_train.columns, lr.coef_)))

#heatmap
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.9])
cmax=np.abs(lr.coef_).max()
cax=ax.matshow(np.reshape(lr.coef_,[-1,1]),cmap='seismic',aspect='1',vmax=cmax,vmin=-1.0*cmax)
fig.colorbar(cax)
ax.set_yticks(list(range(len(df_train.columns[:-1]))))
ax.set_yticklabels(df_train.columns[:-1])
ax.set_xticks([])
ax.set_xlabel('Logistic Regression Betas')
#plt.show()