import numpy as np
from scipy.optimize import newton

def evaluate_function(func_str, x_values):
    func = lambda x: eval(func_str)
    return [func(x) for x in x_values]

def find_approximate_root(func_str, x_range):
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = evaluate_function(func_str, x_values)
    min_y_index = np.argmin(np.abs(y_values))
    return x_values[min_y_index]

def calculate_absolute_error(approx_root, func_str):
    func = lambda x: eval(func_str)
    numerical_root = newton(func, approx_root)
    return abs(approx_root - numerical_root)

def process_user_input(func_str, x_min, x_max):
    x_range = [float(x_min), float(x_max)]
    x_values = np.linspace(x_range[0], x_range[1], 500)
    y_values = evaluate_function(func_str, x_values)

    approx_root = find_approximate_root(func_str, x_range)
    absolute_error = calculate_absolute_error(approx_root, func_str)
    approx_root_rounded = round(approx_root, 2)
    y_values_rounded = [round(y, 2) for y in y_values]

    return {
        "x_values": x_values.tolist(),
        "y_values": y_values_rounded,
        "approx_root": approx_root_rounded,
        "absolute_error": absolute_error
    }