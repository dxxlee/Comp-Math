import numpy as np

def format_polynomial(coefficients):
    """ Formats a polynomial equation as a human-readable string.
    Parameters:
    - coefficients : list -> List of polynomial coefficients (highest degree first)
     Returns:
    - str -> Formatted polynomial equation string  """
    terms = []  # List to store formatted terms of the polynomial

    for i, coeff in enumerate(coefficients):
        power = len(coefficients) - i - 1  # Determine the exponent of x
        # Ignore very small coefficients (close to zero)
        if abs(coeff) < 1e-10:
            continue
        # Format the polynomial term based on its degree
        if power == 0:
            term = f"{coeff:.4f}"  # Constant term
        elif power == 1:
            term = f"{coeff:.4f}x"  # Linear term (e.g., 2.34x)
        else:
            term = f"{coeff:.4f}x^{power}"  # Higher-degree term (e.g., 3.12x^2)

        terms.append(term)  # Add formatted term to the list

    # Join terms into a single equation string and clean up formatting
    return "y = " + " + ".join(terms).replace("+ -", "- ")


def fit_polynomial(x_data, y_data, degree=3):
    """ Fits a polynomial of the given degree to data points (x_data, y_data). """

    coefficients = np.polyfit(x_data, y_data, deg=degree)  # Compute polynomial coefficients
    polynomial = np.poly1d(coefficients)  # Create polynomial function

    formatted_equation = format_polynomial(coefficients)  # Format equation string

    # Generate x values for plotting the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)  
    y_fit = polynomial(x_fit)  # Compute corresponding y values

    return {
        'coefficients': coefficients.tolist(),  # Polynomial coefficients
        'equation': formatted_equation,  # Human-readable polynomial equation
        'x_fit': x_fit.tolist(),  # X values for curve plotting
        'y_fit': y_fit.tolist(),  # Y values for curve plotting
        'x_data': x_data,  # Original data points
        'y_data': y_data
    }
