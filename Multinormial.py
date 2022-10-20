import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df_train=pd.read_csv("/Users/ningduan/Python/MachineLearning/LogisticRegression/heart_train.csv")
df_test=pd.read_csv("/Users/ningduan/Python/MachineLearning/LogisticRegression/heart_test.csv")
#print(df_train)
#target would be y
#input dataset
X_train=df_train[df_train.columns[:-1]].to_numpy()
X_test=df_train[df_test.columns[:-1]].to_numpy()
#print(X_train)
y_train=df_train[df_train.columns[-1]].to_numpy()
y_test=df_train[df_test.columns[-1]].to_numpy()
#split train dataset to train and validation dataset to run regularization and trade off between accuracy and model complexity
df_train_multi_validation,df_test_multi_validation=train_test_split(df_train,test_size=0.25,random_state=0)

#set up x and y
X_train_multi_validation=df_train_multi_validation[[x for x in df_train.columns if x!='cp']].to_numpy()
y_train_multi_validation=df_train_multi_validation['cp'].to_numpy()
X_test_multi_validation=df_test_multi_validation[[x for x in df_train.columns if x!='cp']].to_numpy()
y_test_multi_validation=df_test_multi_validation['cp'].to_numpy()

#scaler
ss = StandardScaler()# the same thing with z-score transformation
X_train_multi_validation_z = ss.fit_transform(X_train_multi_validation)
X_train_multi_validation_mean = X_train_multi_validation.mean(axis=0)#select object across rows, vertically
X_train_multi_validation_std = X_train_multi_validation.std(axis=0)
X_test_multi_validation_z=(X_test_multi_validation-X_train_multi_validation_mean)/X_train_multi_validation_std

#search over regulation strength and find the best settings that results in the best performance on test validation dataset
models={}
validation_accs=[]
for c in [np.inf,1000,100,10,1,0.1,0.01,0.001,0.0001]:
    print("C="+str(c))
    if c==np.inf:
        mlr=LogisticRegression(max_iter=10000,multi_class='multinomial',penalty='none')
    else:
        mlr=LogisticRegression(max_iter=10000,multi_class='multinomial',penalty='l2',C=c)
    mlr.fit(X_train_multi_validation_z,y_train_multi_validation)
    models[c]=mlr
    #y_train_multi_validation_preds=mlr.predict(X_train_multi_validation_z)
    test_validation_accs=mlr.score(X_test_multi_validation_z,y_test_multi_validation)
    validation_accs.append(test_validation_accs)
    print("Train validation acc: ",mlr.score(X_train_multi_validation_z,y_train_multi_validation))
    print("Test validation acc: ",test_validation_accs)
    print()
#c=0.1 should be a good choice

#offical run the test dataset
X_test_multi=df_test[[x for x in df_test.columns if x!='cp']].to_numpy()
y_test_multi=df_test['cp'].to_numpy()
ss = StandardScaler()# the same thing with z-score transformation
X_test_multi_z=(X_test_multi-X_train_multi_validation_mean)/X_train_multi_validation_std #we are not using scaler for test dataset, we want use regular z-score calculation

#compare accuracy of test validation dataset with test dataset

test_accs=[]
for i, c in enumerate([np.inf,1000,100,10,1,0.1,0.01,0.001,0.0001]):
    print('C='+str(c))
    print("validation test accuracy:", validation_accs[i])
    test_acc = mlr.score(X_test_multi_z,y_test_multi)
    test_accs.append(test_acc)
    print("test accuracy:",test_acc)

#heatmap
fig=plt.figure()
ax=fig.add_axes([0.1,0.2,0.8,0.9])
cmax=np.abs(mlr.coef_).max()
cax=ax.matshow(mlr.coef_.T,cmap='seismic',aspect='1',vmax=cmax,vmin=-1.0*cmax)
fig.colorbar(cax)
ax.set_yticks(list(range(len(df_train.columns[:-1]))))
ax.set_yticklabels([x for x in df_train.columns if x!='cp'])
ax.set_xticks(list(np.unique(y_train_multi)))
ax.set_xticklabels(list(np.unique(y_train_multi)))
ax.set_xlabel('Class')
plt.title('Multinormial regression betas')
#plt.show()
#print(mlr.coef_.T)#transpose