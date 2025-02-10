import numpy as np

def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)  # Нижняя треугольная матрица
    U = A.copy()   # Верхняя треугольная матрица

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    return L, U

def solve_lu(L, U, b):
    n = L.shape[0]
    y = np.zeros(n)
    
    # Прямой ход для Ly = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Обратный ход для Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x