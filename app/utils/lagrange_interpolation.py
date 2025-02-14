def lagrange_interpolation(x, y, x_value):
    n = len(x)
    result = 0

    for i in range(n):
        term = y[i]

        for j in range(n):
            if j != i:
                term *= (x_value - x[j]) / (x[i] - x[j])

        result += term

    return result