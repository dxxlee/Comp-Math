import numpy as np

def format_polynomial(coefficients):
    terms = []
    for i, coeff in enumerate(coefficients):
        power = len(coefficients) - i - 1
        if abs(coeff) < 1e-10:
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
    coefficients = np.polyfit(x_data, y_data, deg=degree)
    polynomial = np.poly1d(coefficients)
    formatted_equation = format_polynomial(coefficients)

    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = polynomial(x_fit)

    return {
        'coefficients': coefficients.tolist(),
        'equation': formatted_equation,
        'x_fit': x_fit.tolist(),
        'y_fit': y_fit.tolist(),
        'x_data': x_data,
        'y_data': y_data
    }