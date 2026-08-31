import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Advertising.csv")
X=data["TV"].values
Y=data["sales"].values
degree=int(input("Enter degree of Equation:"))
X_poly=np.column_stack([X**i for i in range(degree+1)])

beta=np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ Y
Y_pred=X_poly @ beta

print("\nCoefficients\n")
for i in range(degree+1):
    print("b",i,"=",beta[i])
mse=np.mean((Y_pred-Y)**2)
ss_residual=np.sum((Y_pred-Y)**2)
ss_total=np.sum((Y-np.mean(Y))**2)
r2=1-(ss_residual/ss_total)
print("MSE:",mse)
print("R2:",r2)

order=np.argsort(X)
X_sorted=X[order]
Y_pred_sorted=Y_pred[order]
plt.scatter(X,Y,label="Actual")
plt.plot(X_sorted,Y_pred_sorted,label="Polynomial fit")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.legend()
plt.show()