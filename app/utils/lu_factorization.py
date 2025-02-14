import numpy as np

def lu_factorization(A):
    """Performs LU Factorization: A = LU, where L is lower triangular, U is upper triangular."""

    n = A.shape[0]  # Matrix size (assuming A is square)
    L = np.eye(n)  # Initialize L as an identity matrix
    U = A.copy()   # Copy A to transform into U

    for i in range(n):  # Iterate over columns (pivot elements)
        for j in range(i + 1, n):  # Eliminate elements below the pivot
            factor = U[j, i] / U[i, i]  # Compute multiplier
            L[j, i] = factor  # Store multiplier in L
            U[j, i:] -= factor * U[i, i:]  # Update U row

    return L, U  # Return L (lower) and U (upper)

def solve_lu(L, U, b):
    """Solves Ax = b using LU decomposition (Ly = b, then Ux = y)."""

    n = L.shape[0]
    y = np.zeros(n)  # Initialize y for forward substitution

    for i in range(n):  # Forward substitution: Solve Ly = b
        y[i] = b[i] - np.dot(L[i, :i], y[:i])  # Compute y[i]

    x = np.zeros(n)  # Initialize x for backward substitution

    for i in range(n - 1, -1, -1):  # Backward substitution: Solve Ux = y
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]  # Compute x[i]

    return x  # Return solution vector
