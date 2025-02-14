import numpy as np


def booles_rule(func_str, a, b, n):
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4")

    x_symbol = 'x'
    func = lambda x: eval(func_str)

    x = np.linspace(a, b, n + 1)
    try:
        y = np.array([func(xi) for xi in x])
    except Exception as e:
        raise ValueError(f"Error evaluating function: {str(e)}")

    h = (b - a) / n

    integral = (7 * h / 45) * (
            7 * (y[0] + y[-1]) +
            32 * sum(y[1:-1:4]) +
            12 * sum(y[2:-2:4]) +
            14 * sum(y[3:-3:4])
    )

    x_plot = np.linspace(a, b, 200)
    y_plot = np.array([func(xi) for xi in x_plot])

    return {
        'integral': float(integral),
        'x_values': x_plot.tolist(),
        'y_values': y_plot.tolist(),
        'sample_points': {
            'x': x.tolist(),
            'y': y.tolist()
        }
    }