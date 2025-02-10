from flask import Blueprint, render_template
from app.utils.gauss_seidel import gauss_seidel
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


@main.route('/gauss-seidel')
def gauss_seidel_route():
    # Define the system of equations
    A = np.array([[4, 1, 1], [1, 5, 1], [1, 1, 6]], dtype=float)
    b = np.array([12, 15, 10], dtype=float)
    x0 = np.zeros(len(b))

    # Solve using Gauss-Seidel method
    solution = gauss_seidel(A, b, x0, tol=1e-6, max_iter=100)

    # Check if the solution is valid (not None)
    has_solution = solution is not None

    # Pass the solution and the flag to the template
    return render_template('gauss_seidel.html', solution=solution, has_solution=has_solution)

@main.route('/lu-factorization')
def lu_factorization():
    return render_template('lu_factorization.html')

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