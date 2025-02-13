import sympy as sp

def fixed_point_iteration(g_str, x0, tol=1e-6, max_iter=50):
    """
    Fixed-Point Iteration with user-specified g(x).
    g_str: строковое представление g(x), например: "x - (exp(-x) + x**2 - 3)"
    Возвращает: (root, iterations, steps_log)
    """
    x = sp.Symbol('x', real=True)
    try:
        g_expr = sp.sympify(g_str)
    except Exception as e:
        return None, 0, [f"Error parsing g(x): {e}"]

    g = sp.lambdify(x, g_expr, "numpy")
    steps = []
    current_x = x0

    for i in range(max_iter):
        try:
            next_x = g(current_x)
        except OverflowError:
            steps.append(f"Step {i+1}: Overflow encountered.")
            return None, i+1, steps
        except Exception as e:
            steps.append(f"Step {i+1}: Error: {e}")
            return None, i+1, steps

        error_val = abs(next_x - current_x)
        steps.append(f"Step {i+1}: x = {current_x:.6f}, g(x) = {next_x:.6f}, error = {error_val:.6e}")

        if error_val < tol:
            return next_x, i+1, steps

        current_x = next_x

    return None, max_iter, steps

import sympy as sp

def newton_raphson(f_str, x0, tol=1e-6, max_iter=50):
    """Метод Ньютона-Рафсона с автоматическим вычислением производной."""

    x = sp.symbols('x', real=True)

    try:
        f_expr = sp.sympify(f_str, locals={'x': x})

        print(f"DEBUG: Parsed f(x): {f_expr}")  
    except Exception as e:
        return None, 0, [f"Error parsing f(x): {e}"], None

    df_expr = sp.diff(f_expr, x)
    derivative_str = str(df_expr)

    print(f"DEBUG: Computed f'(x): {derivative_str}")

    f = sp.lambdify(x, f_expr, "numpy")
    df = sp.lambdify(x, df_expr, "numpy")

    steps = []
    current_x = x0

    for i in range(max_iter):
        f_val = f(current_x)
        df_val = df(current_x)

        if abs(df_val) < 1e-14:
            steps.append(f"Step {i+1}: derivative near zero at x = {current_x:.6f}.")
            return None, i+1, steps, derivative_str

        next_x = current_x - f_val / df_val
        error_val = abs(next_x - current_x)

        steps.append(
            f"Step {i+1}: x = {current_x:.6f}, f(x) = {f_val:.6f}, "
            f"f'(x) = {df_val:.6f}, next_x = {next_x:.6f}, error = {error_val:.6e}"
        )

        if abs(f_val) < tol:
            return next_x, i+1, steps, derivative_str

        current_x = next_x

    return None, max_iter, steps, derivative_str
