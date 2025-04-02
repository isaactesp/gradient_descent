# -*- coding: utf-8 -*-
"""
In this script, we can see that the method converges with data that has a high correlation
coefficient. We can see that now the Perason's coefficient is much higher
"""

import numpy as np
import matplotlib.pyplot as plt
import gradientDescent as fp

#To check the analytical solution
from sklearn.linear_model import LinearRegression


#In this script we are studying the relation between the years of study of a person with the
#salary this person is earning per year(in euros and x1000)

years_of_study=np.array([10, 12, 16, 14, 18, 8, 12, 11])
salary=np.array([30, 35, 50, 40, 60, 25, 33, 31])



#We see the correlation between x and y with the Pearson coefficient
correlation1=np.corrcoef(years_of_study,salary)[0,1]
print(f"Correlation between the amount of years studied and the salary of the person: {correlation1}")


#Parameters of the method,lr has been found with the script findingLR.py
lr=0.005
maxIterations=10000
tol=0.01
decreasingCoeff='const'

w_opt,b_opt,i=fp.regression_by_gradientDescent(years_of_study,salary,lr, decreasingCoeff,tol,maxIterations)

print(f'\nOptimus w and b obtained after the gradient descent:\n w: {w_opt} b:{b_opt}')
print(f'Obtaied in: {i} iterations')


y_pred=w_opt*years_of_study+b_opt

MeanSquareError=fp.MSE(y_pred,salary)
print(f'Mean square error of the regression obtained by the gradient descent: {MeanSquareError}')

#Calculate the analytical solution
model=LinearRegression()
#put the independent variable, years of study, in matrix format and calculate the model, the dependent variable can stay in array format
model.fit(years_of_study.reshape(years_of_study.size,1),salary)
w_analytical=model.coef_[0]
b_analytical=model.intercept_
salary_pred=model.predict(years_of_study.reshape(years_of_study.size,1))
print(f"The analytical solution is: w={w_analytical} b={b_analytical}")

#Plot the points
plt.scatter(years_of_study,salary,color='red',label='Data')
#Plot the regression calculated by the gradient descent
plt.plot(years_of_study,y_pred,color='blue',label='Numerical regression')
#Plot the regression calculated by python, the analytical solution
plt.plot(years_of_study,salary_pred,color='green',label='Analytical regression')
plt.xlabel('Years of study')
plt.ylabel('Salary')
plt.legend()
plt.show()
#We can see that are so close that we can't even distinguish one from the other




