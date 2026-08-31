import numpy as np
import pandas as pd

data=pd.read_csv("logistic.csv")
X=data[["Age","Salary"]].values
Y=data["Purchased"].values

np.random.seed(42)
indices=np.random.permutation(len(X))
train_size=int(0.8*len(X))
train_idx=indices[:train_size]
test_idx=indices[train_size:]
X_train=X[train_idx]
Y_train=Y[train_idx]
X_test=X[test_idx]
Y_test=Y[test_idx]

mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)

X_train=(X_train-mean)/std
X_test=(X_test-mean)/std

X_train=np.c_[np.ones(len(X_train)),X_train]
X_test=np.c_[np.ones(len(X_test)),X_test]
weights=np.zeros(X_train.shape[1])
lr=0.01
epochs=10000
def sigmoid(z):
    return 1/(1+np.exp(-z))
for i in range(epochs):
    z=np.dot(X_train,weights)
    probability=sigmoid(z)
    error=probability-Y_train
    gradient=np.dot(X_train.T,error)/len(Y_train)
    weights=weights-(lr*gradient)
probability=sigmoid(np.dot(X_test,weights))
Y_pred=(probability>=0.5).astype(int)

TP=np.sum((Y_pred==1)&(Y_test==1))
TN=np.sum((Y_pred==0)&(Y_test==0))
FP=np.sum((Y_pred==1)&(Y_test==0))
FN=np.sum((Y_pred==0)&(Y_test==1))

acc=(TP+TN)/(TP+TN+FP+FN)
pre=TP/(TP+FP)
recall=TP/(TP+FN)
f1=(2*pre*recall)/(pre+recall)

print("Accuracy:",acc)
print("Precision:",pre)
print("Recall:",recall)
print("F1:",f1)
