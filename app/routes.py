from flask import Blueprint, render_template

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
def gauss_seidel():
    return render_template('gauss_seidel.html')

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