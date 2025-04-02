# -*- coding: utf-8 -*-
"""
In this script, we can see that the method is useless with data that has a low correlation
coefficient. Doesn't have sense calculate a linear regression with no correlated data.
We are using the Pearson's correlation coefficient, being 0 no linear relation 
and 1 the best positive linear relation.

"""

import numpy as np
import matplotlib.pyplot as plt
import gradientDescent as fp


#I obtained some data from the relation between the phosphorus of the land(x) and the phosphorus of the plant growing(y)
phosphorus_land=np.array([54,23,19,34,24,65,44,31,29,58,37,46,50,44,56,36,31])

phosphorus_plant=np.array([64,60,71,61,54,77,81,93,93,51,76,96,77,93,95,54,99])

#We see the correlation between x and y with the Pearson coefficient
correlation=np.corrcoef(phosphorus_plant,phosphorus_land)[0,1]
print("Correlation between the phosphorus of the land and the phosphorus of the plant: ",correlation)

#Parameters of the method
lr=0.001#I have tried many lr thanks to the test4 and all of them have high mse
maxIterations=10000
tol=0.01
decreasingCoeff='const'

w_opt,b_opt,i=fp.regression_by_gradientDescent(phosphorus_land,phosphorus_plant,lr,decreasingCoeff,tol,maxIterations)

print(f'\nOptimus w and b for the regression y_prediction=wx+b obtained after the gradient descent: {w_opt},{b_opt}\n')
print(f'Obtaied in: {i} iterations')

#Calculate the values of y_predicted
y_pred=w_opt*phosphorus_land+b_opt

MeanSquareError=fp.MSE(y_pred,phosphorus_plant)
print(f'Mean square error of the regression obtained by the gradient descent: {MeanSquareError}')

#Plot the results
plt.scatter(phosphorus_land,phosphorus_plant,color='red',label='Data')
plt.plot(phosphorus_land,y_pred,color='blue',label='Linear regression')
plt.xlabel('Phosphorus of the land')
plt.ylabel('Phosphorus of the plant')
plt.legend()
plt.show()