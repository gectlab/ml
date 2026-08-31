import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("svm.csv")
X=data[["X1","X2"]].values
Y=data["Y"].values

w=np.zeros(2)
b=0
lr=0.001
epochs=10000

for epoch in range(epochs):
    for i in range(len(X)):
        condition=Y[i]*(np.dot(X[i],w)+b)
        if condition>=1:
            dw=w
            db=0
        else:
            dw=w-Y[i]*X[i]
            db=-Y[i]
        w=w-lr*dw
        b=b-lr*db
print("\nWeights:",w)
print("Bias:",b)
w_norm=np.sqrt(np.sum(w**2))
margin_width=2/w_norm
print("Margin width:",margin_width)

x=np.linspace(0,10,100)
plt.scatter(X[Y==1,0],X[Y==1,1])
plt.scatter(X[Y==-1,0],X[Y==-1,1])
plt.plot(x,-(w[0]*x+b)/w[1])
plt.plot(x,-(w[0]*x+b-1)/w[1],"--")
plt.plot(x,-(w[0]*x+b+1)/w[1],"--")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()