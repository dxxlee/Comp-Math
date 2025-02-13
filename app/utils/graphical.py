import numpy as np
from scipy.optimize import newton

def evaluate_function(func_str, x_values):
    """Evaluate the user-defined function f(x) for given x values."""
    func = lambda x: eval(func_str)
    return [func(x) for x in x_values]

def find_approximate_root(func_str, x_range):
    """Find an approximate root using the graphical method."""
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = evaluate_function(func_str, x_values)
    min_y_index = np.argmin(np.abs(y_values))
    return x_values[min_y_index]

def calculate_absolute_error(approx_root, func_str):
    """Calculate absolute error compared to the root found using a numerical method."""
    func = lambda x: eval(func_str)
    numerical_root = newton(func, approx_root)  # Use Newton's method for numerical root finding
    return abs(approx_root - numerical_root)

def process_user_input(func_str, x_min, x_max):
    """Process user input and return results."""
    x_range = [float(x_min), float(x_max)]
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = evaluate_function(func_str, x_values)

    # Find approximate root
    approx_root = find_approximate_root(func_str, x_range)

    # Calculate absolute error
    absolute_error = calculate_absolute_error(approx_root, func_str)

    # Round values to 2 decimal places (only for display purposes)
    approx_root_rounded = round(approx_root, 2)
    y_values_rounded = [round(y, 2) for y in y_values]

    return {
        "x_values": x_values.tolist(),
        "y_values": y_values_rounded,
        "approx_root": approx_root_rounded,
        "absolute_error": absolute_error  # Pass the raw value without rounding
    }