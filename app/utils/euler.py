import numpy as np
import sympy as sp
import re


def parse_equation(equation_str):
    # Define mathematical symbols
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    derivative = sp.Symbol('derivative')  # Represent dy/dx

    # Clean up the equation string
    equation_str = equation_str.replace(" ", "")

    # Replace dy/dx with our derivative symbol
    equation_str = re.sub(r'dy/dx', 'derivative', equation_str)

    try:
        # Split the equation into left side and right side by "="
        if '=' in equation_str:
            lhs_str, rhs_str = equation_str.split('=')
        else:
            raise ValueError("Equation must contain '='")

        # Parse both sides of the equation
        local_dict = {'x': x, 'y': y, 'derivative': derivative}
        lhs = sp.sympify(lhs_str, locals=local_dict)
        rhs = sp.sympify(rhs_str, locals=local_dict)

        # Create equation: lhs = rhs
        equation = sp.Eq(lhs, rhs)

        # Solve for the derivative
        solved = sp.solve(equation, derivative)
        if not solved:
            raise ValueError("Could not solve equation for dy/dx")

        # Get the first solution (there should only be one for these types of equations)
        derivative_expr = solved[0]

        #  return derivative expression, mathematical x and y symbols
        return derivative_expr, x, y

    except Exception as e:
        raise ValueError(f"Error parsing equation: {str(e)}")


def euler_method(f, x0, y0, h, n):
    """
    Implements Euler's Method for solving first-order ordinary differential equations (ODEs).
    
    Parameters:
    - f: function        -> The derivative function dy/dx = f(x, y)
    - x0: float          -> Initial x value (starting point)
    - y0: float          -> Initial y value (function value at x0)
    - h: float           -> Step size (increment in x)
    - n: int             -> Number of steps (iterations)

    Returns:
    - x: list            -> List of x values
    - y: list            -> Corresponding y values computed using Euler's method
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
        try:
            slope = float(f(x[i], y[i]))  # Ensure we get a numerical value
            y[i + 1] = y[i] + h * slope  # Compute next y using Euler's formula
        except Exception as e:
            raise ValueError(f"Error computing step {i}: {str(e)}")

    # Convert numpy arrays to lists for easier handling in external functions
    return x.tolist(), y.tolist()