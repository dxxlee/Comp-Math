import numpy as np

def euler_method(f, x0, y0, h, n):
    """
    Implements Euler's Method for solving first-order ordinary differential equations (ODEs).
    
    Parameters:
    - f : function        -> The derivative function dy/dx = f(x, y)
    - x0 : float          -> Initial x value (starting point)
    - y0 : float          -> Initial y value (function value at x0)
    - h : float           -> Step size (increment in x)
    - n : int             -> Number of steps (iterations)

    Returns:
    - x : list            -> List of x values
    - y : list            -> Corresponding y values computed using Euler's method
    """

    # Initialize arrays to store x and y values
    x = np.zeros(n + 1)  # Array to store x values
    y = np.zeros(n + 1)  # Array to store corresponding y values

    # Set initial conditions
    x[0] = x0  # Starting x value
    y[0] = y0  # Corresponding y value (initial condition)

    # Perform Euler's method iterations
    for i in range(n):
        x[i + 1] = x[i] + h  # Compute next x value (step forward by h)
        y[i + 1] = y[i] + h * f(x[i], y[i])  # Compute next y using Euler's formula:

    # Convert numpy arrays to lists for easier handling in external functions
    return x.tolist(), y.tolist()
