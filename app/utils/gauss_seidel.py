import numpy as np


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    """
    Solves a system of linear equations using the Gauss-Seidel iterative method.

    Parameters:
        A (numpy array): Coefficient matrix of the system.
        b (numpy array): Right-hand side vector of the system.
        x0 (numpy array): Initial guess for the solution.
        tol (float): Tolerance for stopping criterion (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        numpy array: Solution vector if convergence is achieved.
        None: If the method fails to converge within max_iter iterations.
    """
    n = len(b)  # Number of equations
    x = np.array(x0, dtype=float)  # Convert initial guess to a numpy array

    for iteration in range(max_iter):
        x_new = np.copy(x)  # Create a copy of the current solution vector

        # Perform Gauss-Seidel iteration
        for i in range(n):
            summation = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_new[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - summation) / A[i][i]

        # Check the stopping condition (infinity norm of the difference)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return x_new

        x = x_new  # Update the solution vector for the next iteration
        print(f"Iteration {iteration + 1}: {x}")

    print("Warning: Maximum iterations reached without convergence.")
    return None  # Return None if the method fails to converge