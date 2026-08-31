import numpy as np
import pandas as pd

data=pd.read_csv("Advertising.csv")
X=data["TV"].values
Y=data["sales"].values
n=len(X)
sum_x=np.sum(X)
sum_y=np.sum(Y)
sum_xy=np.sum(X*Y)
sum_x2=np.sum(X**2)

b1=(n*sum_xy-(sum_x*sum_y))/(n*sum_x2-(sum_x**2))
b0=(sum_y-(b1*sum_x))/n

Y_pred=b0+(b1*X)

mse=np.mean((Y_pred-Y)**2)
print("MSE:",mse)

ss_residual=np.sum((Y_pred-Y)**2)
ss_total=np.sum((Y-np.mean(Y))**2)
r2=1-(ss_residual/ss_total)
print("R2:",r2)
print("Slope:",b1)
print("Intercept:",b0)
