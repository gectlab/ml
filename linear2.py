import numpy as np
import pandas as pd

data=pd.read_csv("Advertising.csv")
X=data["TV"].values
Y=data["sales"].values
X_mean=np.mean(X)
X_std=np.std(X)
X_scaled=(X-X_mean)/X_std
n=len(X)
b0=0;b1=0

lr=0.01
epoche=10000

for i in range(epoche):
    Y_pred=b0+(b1*X_scaled)
    error=Y_pred-Y
    db0=(2/n)*np.sum(error)
    db1=(2/n)*np.sum(error*X_scaled)
    b0=b0-(lr*db0)
    b1=b1-(lr*db1)
Y_pred=b0+(b1*X_scaled)
mse=np.mean((Y_pred-Y)**2)
ss_residual=np.sum((Y_pred-Y)**2)
ss_total=np.sum((Y-np.mean(Y))**2)
r2=1-(ss_residual/ss_total)
print("MSE:",mse)
print("R2:",r2)
print("Slope:",b1)
print("Intercept:",b0)