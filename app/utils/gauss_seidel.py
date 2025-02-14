import numpy as np


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.array(x0, dtype=float)

    for iteration in range(max_iter):
        x_new = np.copy(x)

        # Perform Gauss-Seidel iteration
        for i in range(n):
            summation = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_new[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - summation) / A[i][i]

        # Check the stopping condition
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return x_new

        x = x_new
        print(f"Iteration {iteration + 1}: {x}")

    print("Warning: Maximum iterations reached without convergence.")
    return None