# -*- coding: utf-8 -*-
"""
In this script I find the relation between the amount of data and the interations needed to converge.
The data analised follows a normal distribution with mean 0.5 and it's between (0,1), so will have a 
higher correlation.
We can notice that some times the decreasingCoeff is very useful. 
"""

import numpy as np
import matplotlib.pyplot as plt
import gradientDescent as fp

#parameters of the method
lr=0.1
maxIterations=10000 
tol=0.01

#In this test we can prove that when we are starting with a higher learning rate, is better to use the decreasing 
#coefficient 
decreasingCoeff=lr/5

#Where I'll save the values
data_sizes=[]
iterations_needed=[]


#Starting in a minimum amount of points 
for size in range(5,106):
    #To have a high Pearson's coefficient of correlation, generate numbers with normal distribution and liner correlated
    x_values=5*np.random.rand(size)
    #I add a noise because if not the problem would be trivial
    y_values=3*x_values+9*np.random.rand()
    
    w_opt, b_opt, iterations = fp.regression_by_gradientDescent(x_values, y_values, lr,decreasingCoeff, tol, maxIterations)
    
    data_sizes.append(size)
    iterations_needed.append(iterations)
    
    
    if size==105:
        print("Data processed")

#Plot the results
plt.plot(data_sizes, iterations_needed)
plt.xlabel('Amount of data')
plt.ylabel('Iterations to converge')
plt.title('Relation between the amount of data and the convergence')
plt.grid(True)





