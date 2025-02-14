def lagrange_interpolation(x, y, x_value):
    """    Computes the Lagrange interpolation polynomial at a given x_value.
    Parameters:
    - x : list -> List of known x-coordinates (data points)
    - y : list -> Corresponding y-coordinates (function values at x)
    - x_value : float -> The x-value at which interpolation is needed
    
    Returns:
    - float -> Interpolated y-value at x_value    """

    n = len(x)  # Number of data points
    result = 0  # Initialize result

    # Iterate over each data point to construct Lagrange basis polynomials
    for i in range(n):
        term = y[i]  # Start with y[i] as the base term

        # Compute the Lagrange basis polynomial L_i(x)
        for j in range(n):
            if j != i:  # Skip when j == i (avoid division by zero)
                term *= (x_value - x[j]) / (x[i] - x[j])  # Compute the product
        result += term  # Add each Lagrange term to the result
        
    return result  # Return the final interpolated value
