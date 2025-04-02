# Linear Regression with Gradient Descent

This project implements **linear regression** using a **custom gradient descent algorithm** from scratch in Python. The main goal is to approximate the linear relationship between two variables by minimizing the **Mean Square Error (MSE)**. The project is built to understand the inner mechanics of gradient descent and how learning rate, tolerance, and data size affect convergence.

---

## Specification

- `gradientDescent.py`: Core module implementing the gradient descent algorithm and derivative functions.
- `findigLR.py`: Script to empirically find the best learning rate for a dataset.
- `test1.py`: Shows failure with low-correlation data.
- `test2.py`: Analyzes how dataset size affects convergence time.
- `test3.py`: Compares the custom solution with scikit-learn's analytical regression.


---

## Features

- Full implementation of gradient descent from scratch (no ML libraries).
- Automatic computation of numerical partial derivatives.
- Support for:
  - Constant or decreasing learning rates.
  - Convergence based on gradient magnitude.
- Evaluation of performance using MSE.
- Empirical analysis tools (learning rate tuning, convergence behavior).
