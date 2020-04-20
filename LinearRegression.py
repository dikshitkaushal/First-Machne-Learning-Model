# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:36:49 2020

@author: DELL
"""
"importing libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"importing the dataset:"
dataset=pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"Splitting the Dataset into Training set and Test Set"
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 
"Training Simple Regression model on the training set"
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

"Predicting the Test Set results"
ypred=regressor.predict(x_test)

"Visualising the training set results"
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

"Visualising the test set results"
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary Vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
