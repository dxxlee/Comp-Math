from flask import Blueprint, render_template, request, jsonify
from sympy import euler

from app.utils.gauss_seidel import gauss_seidel
from app.utils.lu_factorization import lu_factorization, solve_lu
from app.utils.polynomial_curve_fitting import fit_polynomial
from app.utils.lagrange_interpolation import lagrange_interpolation
from app.utils.euler import euler_method
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

@main.route('/lu-factorization', methods=['GET', 'POST'])
def lu_factorization_route():
    result = None
    error = None
    matrix_size = 3  # Размер матрицы по умолчанию

    if request.method == 'POST':
        try:
            # Получаем размер матрицы от пользователя
            matrix_size = int(request.form['matrix_size'])

            # Считываем матрицу A
            A = []
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = float(request.form[f'A_{i}_{j}'])
                    row.append(value)
                A.append(row)

            # Считываем вектор b
            b = [float(request.form[f'b_{i}']) for i in range(matrix_size)]

            # Применяем LU-разложение
            L, U = lu_factorization(np.array(A))
            x = solve_lu(L, U, np.array(b))
            check = np.dot(A, x)

            result = {
                'L': L.tolist(),
                'U': U.tolist(),
                'solution': x.tolist(),
                'check': check.tolist()
            }

        except Exception as e:
            error = str(e)

    return render_template('lu_factorization.html', result=result, error=error, matrix_size=matrix_size)

@main.route('/polynomial-curve-fitting', methods=['GET', 'POST'])
def polynomial_fit_route():
    result = None
    error = None

    if request.method == 'POST':
        try:
            # Get the number of points
            num_points = int(request.form['num_points'])

            # Parse input data
            x_data = [float(request.form[f'x_{i}']) for i in range(num_points)]
            y_data = [float(request.form[f'y_{i}']) for i in range(num_points)]
            degree = int(request.form['degree'])

            # Fit the polynomial
            result = fit_polynomial(x_data, y_data, degree)

        except Exception as e:
            error = str(e)

    return render_template('polynomial_curve_fitting.html', result=result, error=error)

@main.route('/lagrange-interpolation', methods=['GET', 'POST'])
def lagrange_interpolation_route():
    result = None
    error = None

    if request.method == 'POST':
        try:
            # Get the number of points
            num_points = int(request.form['num_points'])

            # Parse input data
            x_data = [float(request.form[f'x_{i}']) for i in range(num_points)]
            y_data = [float(request.form[f'y_{i}']) for i in range(num_points)]
            x_value = float(request.form['x_value'])

            # Perform Lagrange interpolation
            interpolated_value = lagrange_interpolation(x_data, y_data, x_value)
            result = {
                'x_value': x_value,
                'interpolated_value': interpolated_value
            }

        except Exception as e:
            error = str(e)

    return render_template('lagrange_interpolation.html', result=result, error=error)

@main.route('/booles-rule')
def booles_rule():
    return render_template('booles_rule.html')

@main.route('/euler-method', methods=['GET', 'POST'])
def euler_view():  # Renamed from euler to euler_view to avoid conflicts
    if request.method == 'POST':
        try:
            data = request.get_json()
            x0 = float(data['x0'])
            y0 = float(data['y0'])
            h = float(data['h'])
            n = int(data['n'])
            equation = data['equation']

            f = lambda x, y: eval(equation)
            x_values, y_values = euler_method(f, x0, y0, h, n)

            return jsonify({
                'x_values': x_values,
                'y_values': y_values
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return render_template('euler_method.html')
