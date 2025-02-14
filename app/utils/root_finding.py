import sympy as sp

def fixed_point_iteration(g_str, x0, tol=1e-6, max_iter=50):
    """ Fixed-Point Iteration for solving x = g(x). """

    x = sp.Symbol('x', real=True)  # Define symbolic variable
    try:
        g_expr = sp.sympify(g_str)  # Convert g(x) string to sympy expression
    except Exception as e:
        return None, 0, [f"Error parsing g(x): {e}"]  # Handle parsing errors

    g = sp.lambdify(x, g_expr, "numpy")  # Convert g(x) to numerical function
    steps = []  # Store iteration steps
    current_x = x0  # Initial guess

    for i in range(max_iter):
        try:
            next_x = g(current_x)  # Compute next x
        except OverflowError:
            steps.append(f"Step {i+1}: Overflow encountered.")
            return None, i+1, steps  # Handle numerical overflow
        except Exception as e:
            steps.append(f"Step {i+1}: Error: {e}")
            return None, i+1, steps  # Handle other numerical errors

        error_val = abs(next_x - current_x)  # Compute error
        steps.append(f"Step {i+1}: x = {current_x:.6f}, g(x) = {next_x:.6f}, error = {error_val:.6e}")

        if error_val < tol:  # Convergence check
            return next_x, i+1, steps

        current_x = next_x  # Update x for next iteration

    return None, max_iter, steps  # Return failure if max iterations reached


def newton_raphson(f_str, x0, tol=1e-6, max_iter=50):
    """ Newton-Raphson Method with automatic derivative computation. """
    
    x = sp.symbols('x', real=True)  # Define symbolic variable

    try:
        f_expr = sp.sympify(f_str, locals={'x': x})  # Convert f(x) string to sympy expression
    except Exception as e:
        return None, 0, [f"Error parsing f(x): {e}"], None  # Handle parsing errors
    df_expr = sp.diff(f_expr, x)  # Compute derivative f'(x)
    derivative_str = str(df_expr)  # Convert derivative to string
    f = sp.lambdify(x, f_expr, "numpy")  # Convert f(x) to numerical function
    df = sp.lambdify(x, df_expr, "numpy")  # Convert f'(x) to numerical function
    steps = []  # Store iteration steps
    current_x = x0  # Initial guess

    for i in range(max_iter):
        f_val = f(current_x)  # Compute f(x)
        df_val = df(current_x)  # Compute f'(x)

        if abs(df_val) < 1e-14:  # Avoid division by near-zero derivatives
            steps.append(f"Step {i+1}: derivative near zero at x = {current_x:.6f}.")
            return None, i+1, steps, derivative_str

        next_x = current_x - f_val / df_val  # Newton's update formula
        error_val = abs(next_x - current_x)  # Compute error
        steps.append(
            f"Step {i+1}: x = {current_x:.6f}, f(x) = {f_val:.6f}, "
            f"f'(x) = {df_val:.6f}, next_x = {next_x:.6f}, error = {error_val:.6e}"
        )
        if abs(f_val) < tol:  # Convergence check
            return next_x, i+1, steps, derivative_str

        current_x = next_x  # Update x for next iteration

    return None, max_iter, steps, derivative_str  # Return failure if max iterations reached
