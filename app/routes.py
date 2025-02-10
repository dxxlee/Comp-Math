from flask import Blueprint, render_template, request
from app.utils.gauss_seidel import gauss_seidel
from app.utils.lu_factorization import lu_factorization, solve_lu
import numpy as np

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('base.html')

@main.route('/graphical-method')
def graphical_method():
    return render_template('graphical_method.html')

@main.route('/root-finding')
def root_finding():
    return render_template('root_finding.html')


@main.route('/gauss-seidel', methods=['GET', 'POST'])
def gauss_seidel_route():
    result = None
    error = None
    matrix_size = 3  # Default matrix size

    if request.method == 'POST':
        try:
            # Get matrix size from the form
            matrix_size = int(request.form['matrix_size'])

            # Parse coefficient matrix A
            A = []
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = float(request.form[f'A_{i}_{j}'])
                    row.append(value)
                A.append(row)

            # Parse right-hand side vector b
            b = [float(request.form[f'b_{i}']) for i in range(matrix_size)]

            # Parse initial guess x0
            x0 = [float(request.form[f'x0_{i}']) for i in range(matrix_size)]

            # Parse additional parameters
            tol = float(request.form['tol'])
            max_iter = int(request.form['max_iter'])

            # Solve using Gauss-Seidel method
            solution = gauss_seidel(np.array(A), np.array(b), np.array(x0), tol=tol, max_iter=max_iter)

            if solution is not None:
                result = f"Solution: {solution}"
            else:
                result = "No solution found. The method did not converge."

        except Exception as e:
            error = str(e)

    return render_template('gauss_seidel.html', result=result, error=error, matrix_size=matrix_size)

@main.route('/lu_factorization', methods=['GET', 'POST'])
def lu_factorization_route():
    result = None
    if request.method == 'POST':
        try:
            # Получаем данные от пользователя
            A_input = request.form.get('matrix_A')
            b_input = request.form.get('vector_b')

            # Преобразуем строковый ввод в массивы numpy
            A = np.array(eval(A_input), dtype=float)
            b = np.array(eval(b_input), dtype=float)

            # Вычисляем LU-разложение и решение системы
            L, U = lu_factorization(A)
            x = solve_lu(L, U, b)
            check = np.dot(A, x)

            result = {
                'L': L.tolist(),
                'U': U.tolist(),
                'solution': x.tolist(),
                'check': check.tolist()
            }
        except Exception as e:
            result = {'error': str(e)}

    return render_template('lu_factorization.html', result=result)

@main.route('/polynomial-curve-fitting')
def polynomial_curve_fitting():
    return render_template('polynomial_curve_fitting.html')

@main.route('/lagrange-interpolation')
def lagrange_interpolation():
    return render_template('lagrange_interpolation.html')

@main.route('/euler-method')
def euler_method():
    return render_template('euler_method.html')

@main.route('/booles-rule')
def booles_rule():
    return render_template('booles_rule.html')