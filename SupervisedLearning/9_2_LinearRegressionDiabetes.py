from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np

diabetes = load_diabetes()

diabetes_X, diabetes_y = load_diabetes(return_X_y = True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20] 
diabetes_X_test = diabetes_X[-20:]  #retrieve the last 20 records

diabetes_y_train = diabetes_y[:-20] 
diabetes_y_test = diabetes_y[-20:] 

lin_reg = LinearRegression()

lin_reg.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = lin_reg.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("mse: ", mse)
r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print("r2: ", r2)

plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")
plt.plot(diabetes_X_test, diabetes_y_pred, color = "blue")


