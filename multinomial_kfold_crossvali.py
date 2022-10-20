#import python modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
#load csv with patient demographic information
df=pd.read_csv('/Users/ningduan/Python/MachineLearning/LogisticRegression/heart.csv')
#get a list of input variable (i.e. feature) names
features=[x for x in df.columns if x!='cp']
#create a training and test data arrays for the validation data split for the multinomial regression model
X=df[features].to_numpy()
y=df['cp'].to_numpy()
#create k-fold data splitter
k_fold=KFold(n_splits=5)#the higher the better but it takes time
#Grid search over regularization strengths(c) to dind the setting that reults in the best performance ont he test validation solit
models={}
test_accs={}
for c in [np.inf,1000,100,10,1,0.1,0.01,0.001,0.0001]:
    i=0
    fold_models=[]
    fold_accs=[]
    for train_index, test_index in k_fold.split(df):
      X_train_multinomial=X[train_index]
      y_train_multinomial=y[train_index]
      X_test_multinomial=X[test_index]
      y_test_multinomial=y[test_index]

      X_train_multinomial_mean=X_train_multinomial.mean(axis=0)
      X_train_multinomial_std=X_train_multinomial.std(axis=0)

      #z-score the input data arrays for this fold
      X_train_multinomial_z=(X_train_multinomial-X_train_multinomial_mean)/X_train_multinomial_std
      X_test_multinomial_z=(X_test_multinomial-X_train_multinomial_mean)/X_train_multinomial_std

      print('C=' +str(c))
      print("fold:" +str(i))
      if c==np.inf:
          mlr=LogisticRegression(multi_class='multinomial',max_iter=10000, penalty='none')
      else:
          mlr =LogisticRegression(multi_class='multinomial',max_iter=10000, penalty='l2',C=c)
      mlr.fit(X_train_multinomial_z,y_train_multinomial)
      fold_models.append(mlr)
      print("Train validation accuracy: ",mlr.score(X_train_multinomial_z,y_train_multinomial))
      test_acc=mlr.score(X_test_multinomial_z,y_test_multinomial)
      print("Test validation accuracy: ", test_acc)
      fold_accs.append(test_acc)
      i +=1
    models[c]=fold_models
    test_accs[c]=fold_accs
    print()
#print the mean perfomance for each regularization strength across the k-folds
print()
for c in [np.inf,1000,100,10,1,0.1,0.01,0.001,0.0001]:
 print('C=' +str(c))
 print('test accuracy: ',np.array(test_accs[c]).mean())