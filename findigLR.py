# -*- coding: utf-8 -*-
"""
In this script I find the relation between the lr of the method and how the mse varies,
by trying many lr in a certain interval. Thanks to this script we can find the 
best lr for a certain data, tolerance, and decreasing coefficient.
"""

import numpy as np
import gradientDescent as fp
import matplotlib.pyplot as plt


# Parameters of the method
maxIterations=20000
tol=0.01
decreasingCoeff='const'


# Where I'll save the values of the different mse depending on the lr
mse_values=[]
#I create an array with all the lr I want to try
num_of_lr=1000
learning_rates=np.linspace(0.001,0.5,num_of_lr)

# Recall: data must be well correlated
x_values=np.array([10, 12, 16, 14, 18, 8, 12, 11])
y_values=np.array([30, 35, 50, 40, 60, 25, 33, 31])

# Try the all the lr in the interval
for i in range(0,num_of_lr):
    
    w_opt, b_opt, iterations = fp.regression_by_gradientDescent(x_values, y_values,learning_rates[i], decreasingCoeff, tol, maxIterations)
    
    y_pred=w_opt*x_values+b_opt
    
    MSE=fp.MSE(y_values,y_pred)
    #For each lr save the mse
    mse_values.append(MSE)


# Now I take the lr with lowest mse
minMSE=min(mse_values)
print("\nHighlight that this results has been obtained with my gradient descent method to calculate linear regressions with the next parameters: ")
print(f"Data: \nx_values(independent): {x_values}\ny_values(dependent): {y_values}\n")
print(f"Tolerance: {tol}\nMax of iterations: {maxIterations}\nDecreasing coefficient: {decreasingCoeff}")
print(f"\nThe lowest mse reached with learning rates taken between {learning_rates[0]} and {learning_rates[999]} has been {minMSE}")
# I save the position in which this lowest mse is and find the lr on that position
index=np.argmin(mse_values)
bestlr=learning_rates[index]
print(f"This mse is reached with a learning rate of: {bestlr}")

# Let see the plot of the mse depending on the lr, choosing the range of values we wanna plot
plt.plot(learning_rates[2:20],mse_values[2:20],color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Square Error')
plt.title('LR-MSE')
plt.show()
# For the data in test3, we can see that the minimum is reached around 0.005









