import numpy as np

def booles_rule(func_str, a, b, n):
    # Boole's Rule (also known as the 4-point Newton-Cotes quadrature rule)
    # is used for numerical integration over an interval [a, b].
    # It requires `n` to be a multiple of 4 for evenly spaced subintervals.

    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4")  # Ensures correct application of Boole's Rule

    # Define the function dynamically from the given string input.
    # This allows users to input a mathematical function as a string (e.g., "np.sin(x)").
    func = lambda x: eval(func_str)

    # Generate `n+1` equally spaced points between `a` and `b`.
    x = np.linspace(a, b, n + 1)

    # Evaluate the function at these points
    try:
        y = np.array([func(xi) for xi in x])  # Compute function values at each `x`
    except Exception as e:
        raise ValueError(f"Error evaluating function: {str(e)}")  # Handle errors in function evaluation

    # Compute the step size `h`
    h = (b - a) / n  # This is the width of each subinterval

    # Apply Boole's Rule formula:
    # Integral ≈ (7h/45) * [ 7(f(a) + f(b)) + 32∑(odd indices) + 12∑(indices 2,6,10...) + 14∑(indices 3,7,11...) ]
    integral = (7 * h / 45) * (
            7 * (y[0] + y[-1]) +  # First and last points, weighted by 7
            32 * sum(y[1:-1:4]) +  # Every 4th point starting from index 1, weighted by 32
            12 * sum(y[2:-2:4]) +  # Every 4th point starting from index 2, weighted by 12
            14 * sum(y[3:-3:4])    # Every 4th point starting from index 3, weighted by 14
    )

    # Generate additional x values for plotting a smoother curve of the function
    x_plot = np.linspace(a, b, 200)  # 200 points for smooth visualization
    y_plot = np.array([func(xi) for xi in x_plot])  # Compute function values for visualization

    # Return results including the computed integral, sample points, and plotting data
    return {
        'integral': float(integral),  # Final numerical integral result
        'x_values': x_plot.tolist(),  # X values for function plot
        'y_values': y_plot.tolist(),  # Corresponding Y values for function plot
        'sample_points': {  # Points used in numerical integration
            'x': x.tolist(),
            'y': y.tolist()
        }
    }
