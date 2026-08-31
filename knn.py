import numpy as np
import pandas as pd

data=pd.read_csv("KNN.csv")
X=data[["Age","Salary"]].values
Y=data["Purchased"].values

np.random.seed(42)
indices=np.random.permutation(len(X))
train_size=int(0.8*len(X))
train_idx=indices[:train_size]
test_idx=indices[train_size:]
X_train=X[train_idx]
X_test=X[test_idx]
Y_train=Y[train_idx]
Y_test=Y[test_idx]

mean=np.mean(X_train,axis=0)
std=np.mean(X_train,axis=0)
X_train=(X_train-mean)/std
X_test=(X_test-mean)/std

def euclidean_distance(point1,point2):
    return np.sqrt(np.sum((point1-point2)**2))
def knn_predict(X_train,Y_train,test_point,k):
    distances=[]
    for i in range(len(X_train)):
        distance=euclidean_distance(test_point,X_train[i])
        distances.append((distance,Y_train[i]))
    distances.sort()
    neighbours=distances[:k]
    classes=[neighbour[1] for neighbour in neighbours]
    prediction=max(set(classes),key=classes.count)
    return prediction
k=10
Y_pred=[]
for test_points in X_test:
    prediction=knn_predict(X_train,Y_train,test_points,k)
    Y_pred.append(prediction)
Y_pred=np.array(Y_pred)

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
