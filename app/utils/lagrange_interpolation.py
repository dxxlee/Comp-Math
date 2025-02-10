def lagrange_interpolation(x, y, x_value):
    """
    Perform Lagrange interpolation to estimate f(x_value).

    Parameters:
        x (list): List of x-coordinates of the data points.
        y (list): List of y-coordinates of the data points.
        x_value (float): The x-value at which to estimate f(x).

    Returns:
        float: The interpolated value of f(x_value).
    """
    n = len(x)  # Number of data points
    result = 0  # Initialize the result

    for i in range(n):
        term = y[i]  # Start with the y-value of the current point

        # Compute the Lagrange basis polynomial L_i(x)
        for j in range(n):
            if j != i:
                term *= (x_value - x[j]) / (x[i] - x[j])

        result += term  # Add the term to the result

    return result