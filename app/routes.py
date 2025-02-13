from flask import Blueprint, render_template, request, jsonify

from app.utils.gauss_seidel import gauss_seidel
from app.utils.lu_factorization import lu_factorization, solve_lu
from app.utils.polynomial_curve_fitting import fit_polynomial
from app.utils.lagrange_interpolation import lagrange_interpolation
from app.utils.euler import euler_method
from app.utils.graphical import process_user_input
from app.utils.root_finding import fixed_point_iteration, newton_raphson
import numpy as np
import sympy as sp


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('home.html')

@main.route('/graphical-method', methods=['GET', 'POST'])
def graphical_method():
    if request.method == 'POST':
        # Get user input from the form
        func_str = request.form.get('func_str')
        x_min = request.form.get('x_min')
        x_max = request.form.get('x_max')

        # Process the input
        try:
            results = process_user_input(func_str, x_min, x_max)
        except Exception as e:
            return render_template('graphical_method.html', error=str(e))

        # Render the template with results
        return render_template(
            'graphical_method.html',
            x_values=results["x_values"],
            y_values=results["y_values"],
            approx_root=results["approx_root"],
            absolute_error=results["absolute_error"]
        )

    # For GET requests, just render the form
    return render_template('graphical_method.html')


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
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.get_json()
            print("Received JSON data:", data)  # Debugging

            # Parse input data
            num_points = int(data['num_points'])
            x_data = [float(data[f'x_{i}']) for i in range(num_points)]
            y_data = [float(data[f'y_{i}']) for i in range(num_points)]
            degree = int(data['degree'])

            # Fit the polynomial
            result = fit_polynomial(x_data, y_data, degree)
            return jsonify(result)
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 400
    return render_template('polynomial_curve_fitting.html')

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

@main.route('/root-finding', methods=['GET', 'POST'])
def root_finding():
    result = {}  
    error = None

    form_data = {
        "function": "",
        "g_function": "",
        "a": 0.0,
        "b": 3.0,
        "tol": 1e-6,
        "max_iter": 50
    }

    if request.method == 'POST':
        try:
            f_str = request.form['function']
            g_str = request.form['g_function']
            a = float(request.form['a'])
            b = float(request.form['b'])
            tol = float(request.form['tol'])
            max_iter = int(request.form['max_iter'])
            method = request.form['method']
            x0 = (a + b) / 2

            form_data.update({
                "function": f_str,
                "g_function": g_str,
                "a": a,
                "b": b,
                "tol": tol,
                "max_iter": max_iter
            })

            x_sym = sp.Symbol('x', real=True)
            reference_root = None
            try:
                sol = sp.nsolve(sp.sympify(f_str), x_sym, x0)
                reference_root = float(sol)
            except:
                reference_root = None

            if "reference_root" not in result:
                result["reference_root"] = reference_root
            if "function_str" not in result:
                result["function_str"] = f_str
            if "a" not in result:
                result["a"] = a
            if "b" not in result:
                result["b"] = b
            if "tol" not in result:
                result["tol"] = tol
            if "max_iter" not in result:
                result["max_iter"] = max_iter
            if "x0" not in result:
                result["x0"] = x0

            if method == "iteration":
                iter_root, iter_count, iter_steps = fixed_point_iteration(g_str, x0, tol=tol, max_iter=max_iter)
                iter_rel_error = (
                    abs((iter_root - reference_root) / reference_root) * 100 if reference_root else None
                )

                result["iteration_method"] = {
                    "root": iter_root,
                    "iterations": iter_count,
                    "steps": iter_steps,
                    "rel_error": iter_rel_error
                }

            if method == "newton":
                newton_root, newton_count, newton_steps, derivative_str = newton_raphson(f_str, x0, tol=tol, max_iter=max_iter)
                newton_rel_error = (
                    abs((newton_root - reference_root) / reference_root) * 100 if reference_root else None
                )

                result["newton_method"] = {
                    "root": newton_root,
                    "iterations": newton_count,
                    "steps": newton_steps,
                    "rel_error": newton_rel_error,
                    "derivative": derivative_str
                }

        except Exception as e:
            error = str(e)

    return render_template("root_finding.html", result=result, error=error, form_data=form_data)
