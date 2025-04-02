# -*- coding: utf-8 -*-
"""
All the documents are documented at the begining of each script.

The goal of this project is, given a dataset of two variables(x,y), calculate a model(linear regression) 
that aproximates the best relation beween one variable and the other(y_predicted=wx+b) with the lowest 
Mean Square Error between y and y_predicted. So we are looking for the  best w and b, knowing that the MSE(w,b) 
is a convex function, with a minima for one w and b. In this case the y will be the dependent variable and x 
the independent one. 

Highlight that the data should have a high correlation to obtain a good result and same amount of values of x and y
"""

import numpy as np

# Just to type the code
from typing import Callable, Tuple, Union


def MSE(y: np.ndarray, y_predicted: np.ndarray)->float:
    # PRE: y is the real values of the sample we are analyzing and yprediction the values, 
    # calculated with the regression(y and yprediction have same size)
    # POST: returns the Mean Square Error between the values of y and ypredicted
    
    # This function is just to see the final result
    
    sample_size=y.size#same as x.size
    mse=(sum((y_predicted-y)**2))/sample_size
    
    return mse

def dfdx(f:Callable[[float,float],float],x:float,y:float)->float:
    # PRE: f is a function of 2 variables x and y, (x,y) is the point we want to evaluate the derivative
    # Highlight that (x,y) can be in n dimensions, so it's a "point" but x,y can have many dimensions
    # POST: returns the forward derivative of f, respect the first variable x, evaluated in (x,y)
    
    dfdx=(f(x+1e-7,y)-f(x,y))/1e-7
    return dfdx

def dfdy(f:Callable[[float,float],float],x:float,y:float)->float:
    # PRE: f is a function of 2 variables x and y, (x,y) is the point we want to evaluate the derivative
    # Highlight that (x,y) can be in n dimensions, so it's a "point" but x,y can have many dimensions
    # POST: returns the forward derivative of f, respect the second variable y, evaluated in (x,y)
    dfdy=(f(x,y+1e-7)-f(x,y))/1e-7
    return dfdy

def regression_by_gradientDescent(x:np.ndarray,y:np.ndarray,lr:float,decreasingCoeff:Union[float,str],tol:float,maxIterations:int)->Tuple[float,float,int]:
    # PRE: x,y are arrays that contains the data we want to studie(they must have same length), lr is the initial learning rate of the method,
    # maxIterations is a maximum of iterations we give to the method just in case tries to diverge, tol is the tolerance of the method
    # decreasingCoeff is the coefficient that determines how the learning rate decrease. It will stay constant if you introduce a string
    # Highlight the lr use to be between 0.001 and 0.5, due to this is a method that to reach a good convergence uses a big amount of iterations
    # POST: returns the values w,b,i where i is the iterations that the method has reach to get gradient<tol and
    # w and b are the values that best aproximate the regression y_predicted=w*x+b, with minimum MSE(y,y_predicted)
    
    def f(w:float,b:float)->float:
        # PRE: w,b are the two variables of the cost function we wanna minimize with the relation y_predicted=wx+b
        # POST: returns the cost function we wanna minimize(MSE between y_predicted and y in this case)
        
        mse=(sum(((x*w+b)-y)**2))/x.size
        return mse
    
    
    # generate an aleatory number to start with the method, but we could reduce the iterations, selecting ones
    # that are adapted to the data
    w=np.random.rand()
    b=np.random.rand()
   
    # about the learning rate of the method, it's very important to choose the correct one depending on the data
    l0=lr
   
    
    # gradient(f(w,b))=(dfdw,dfdb)
    dfdw=dfdx(f,w,b)
    dfdb=dfdy(f,w,b)
    
    iterations=0

    # we need to reach gradient<tol
    while (abs(dfdw) >tol or abs(dfdb)>tol) and iterations<maxIterations:
        # start decreasing w and b with the learning rate 
        w=w-lr*dfdw
        b=b-lr*dfdb
        dfdw=dfdx(f,w,b)
        dfdb=dfdy(f,w,b)
        
         
        if not isinstance(decreasingCoeff,str):
            # we decrease the learning rate with more iterations to converge quicker
            lr=l0/(1+decreasingCoeff*iterations)
       
        iterations+=1
        
        
        
    # The method will show a message in the screen if the method has reached the maxIterations
    if iterations==maxIterations:
        print("The maximum iterations at the gradient method has been reached.")
        print(f"With {x.size} values of data, a initial learning rate of {l0}, and a tolerance of {tol}")
    
    return w,b,iterations

# After all the test, we see that the lr is the most important parameter of this numerical method




