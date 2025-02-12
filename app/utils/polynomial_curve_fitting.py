import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def format_polynomial(coefficients):
    """
    Formats the polynomial equation in a human-readable form.

    Parameters:
        coefficients (list or numpy array): Coefficients of the polynomial.

    Returns:
        str: Formatted polynomial equation.
    """
    terms = []
    for i, coeff in enumerate(coefficients):
        power = len(coefficients) - i - 1
        if abs(coeff) < 1e-10:  # Ignore very small coefficients
            continue
        if power == 0:
            term = f"{coeff:.4f}"
        elif power == 1:
            term = f"{coeff:.4f}x"
        else:
            term = f"{coeff:.4f}x^{power}"
        terms.append(term)
    return "y = " + " + ".join(terms).replace("+ -", "- ")


def fit_polynomial(x_data, y_data, degree=3):
    """
    Fits a polynomial curve to the given data points.
    Parameters:
        x_data (list or numpy array): x-coordinates of the data points.
        y_data (list or numpy array): y-coordinates of the data points.
        degree (int): Degree of the polynomial to fit (default: 3).
    Returns:
        dict: A dictionary containing the fitted polynomial coefficients, equation, and plot data.
    """
    # Fit the polynomial
    coefficients = np.polyfit(x_data, y_data, deg=degree)
    polynomial = np.poly1d(coefficients)
    # Format the polynomial equation
    formatted_equation = format_polynomial(coefficients)
    # Generate smooth x-values for plotting
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = polynomial(x_fit)
    # Return results
    return {
        'coefficients': coefficients.tolist(),
        'equation': formatted_equation,
        'x_fit': x_fit.tolist(),
        'y_fit': y_fit.tolist(),
        'x_data': x_data,
        'y_data': y_data
    }