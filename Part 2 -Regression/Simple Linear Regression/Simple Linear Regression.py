
"""
Simple Linear Regression
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv('Data_salary_exp.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values
print (X)
print (y)


#Splitting into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


"""#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predictng the Test set results
y_pred = regressor.predict(X_test)


#visualisiong the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#visualisiong the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




