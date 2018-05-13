# -*- coding: utf-8 -*-
"""
Decision tree Regression
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv('Salary_Position.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values


#Fitting decision tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


#prediction of new value
y_pred = regressor.predict(6.5)


#visualisiong the Decision tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Decision tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



#visualisiong the decision tree Regression for smoother curve results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('decision tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

