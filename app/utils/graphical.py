import numpy as np
from scipy.optimize import newton

def evaluate_function(func_str, x_values):
    """Evaluates f(x) for a given set of x values."""
    func = lambda x: eval(func_str)  # Convert function string to a lambda function
    return [func(x) for x in x_values]  # Compute function values at x values

def find_approximate_root(func_str, x_range):
    """Finds an approximate root using the graphical method."""
    x_values = np.linspace(x_range[0], x_range[1], 500)  # Generate 500 x values
    y_values = evaluate_function(func_str, x_values)  # Compute function values

    min_y_index = np.argmin(np.abs(y_values))  # Find index where |f(x)| is smallest
    return x_values[min_y_index]  # Return x corresponding to the smallest |f(x)|

def calculate_absolute_error(approx_root, func_str):
    """Computes absolute error by comparing the graphical method with Newton’s method."""
    func = lambda x: eval(func_str)  # Convert function string to a function
    numerical_root = newton(func, approx_root)  # Use Newton's method for accurate root
    return abs(approx_root - numerical_root)  # Compute absolute error

def process_user_input(func_str, x_min, x_max):
    """Processes user input and finds root approximation."""
    x_range = [float(x_min), float(x_max)]  # Convert range to float
    x_values = np.linspace(x_range[0], x_range[1], 500)  # Generate x values
    y_values = evaluate_function(func_str, x_values)  # Compute function values

    approx_root = find_approximate_root(func_str, x_range)  # Approximate root via graph
    absolute_error = calculate_absolute_error(approx_root, func_str)  # Compute error

    return {
        "x_values": x_values.tolist(),  # X values for plotting
        "y_values": [round(y, 2) for y in y_values],  # Rounded Y values
        "approx_root": round(approx_root, 2),  # Rounded approximate root
        "absolute_error": absolute_error  # Absolute error with Newton’s method
    }
