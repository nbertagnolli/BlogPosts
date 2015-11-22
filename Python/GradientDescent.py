__author__ = 'Nicolas Bertagnolli'

import numpy as np
from pylab import *


def f1(x):
    """Evaluates an ellipse

    Args:
        x: A two element numpy array of the point to be evaluated
    Returns:
        float value of an ellipse evaluated at x
    """
    return .5 * (x[0]**2 + 10*x[1]**2)


def gradient_f1(x):
    return np.array([x[0], 10 * x[1]])


def f(x):
    """Evaluate the Rosenbrock function at x

    Args:
        x: a two dimensional numpy array

    Returns:
        A float value representing the Rosenbrock function evaluated
        at x.
    """
    return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def gradient_f(x):
    """Caclulates the gradient of the Rosenbrock function

    Args:
        x: a two dimensional numpy array

    Returns:
        A two element numpy array.  The first element is
        the derivative with respect to x[0] and the second element
        is the derivative with respect to x[1]
    """
    return np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1] - x[0]**2)])


def hessian_f(x):
    """Caclulates the Hessian of the Rosenbrock function

    Args:
        x: a two dimensional numpy array

    Returns:
        A 2 dimensional numpy array (2X2 matrix).  Representing
        the Hessian of the Rosenbrock function.
    """
    return np.matrix([[-400*x[1]+1200*x[0]**2 + 2, -400*x[0]], [-400*x[0], 200]])


def finite_difference(x, func, epsilon):
    """Calculates the finite difference approximation of the gradient of some
    function func at postition x based on epsilon

    Args:
        x: numpy array of position at which to evaluate the finite difference
        func: A method which maps a numpy array to a float to be evaluated
        epsilon: float how much of a difference to examine

    Returns:
        List of finite differences for the gradient of func
    """
    adjust_eps = np.zeros([len(x), ])  # vector to adjust x by epsilon
    df = []                             # Array of finite difference gradient values
    # Steps through each input and calculates the finite difference approximation
    # of the gradient.
    for i in range(0, len(x)):
        adjust_eps[i] = epsilon
        x_plus_epsilon = x + adjust_eps
        x_minus_epsilon = x - adjust_eps
        diff = (func(x_plus_epsilon) - func(x_minus_epsilon)) / (2*epsilon)
        df.append(diff)
        adjust_eps[i] = 0

    return df


def grad_descent(x_0, func, df, search, conv_tol=10e-5, params=[10e-4, .5]):
    """Performs gradient decent on func with initial conditions of x_0

    Args:
        x_0: A two element numpy array holding the initial points
        func: A method which takes in a two element numpy array and returns a float.  This
        method should represent the function that we want to take the gradient of
        df:   A method which takes in a two element numpy array and returns a two element
        numpy array of the gradient of func
        search: Search methdo for the optimal step size.  This method takes four parameters
        all four are two element numpy arrays the first is the current x the second is our change in x
        the third is our change in the function, and the fourth is an alpha and beta parameter for line search
        conv_tol: The convergense tolerance a float
        params: Optional two element numpy array holding search parameters

    Returns:
        x: two element numpy array which is the optimal x point.
        num_iter: integer The number of iterations until convergence
        x_vals: A list of two element numpy arrays filled with each step timepoint
    """
    x = x_0
    num_iter = 0
    x_vals = [x_0]
    while np.linalg.norm(df(x), 2) > conv_tol:
        delta_x = -df(x)
        eta = search(x, delta_x, -delta_x, func, params)
        x = x + eta * delta_x
        num_iter += 1
        x_vals.append(x)

    return x, num_iter, x_vals


def backtracking_line_search(x, delta_x, delta_f, func, params):
    alpha = params[0]
    beta = params[1]
    eta = 1
    while func(x + eta * delta_x) > func(x) + alpha * eta * np.dot(delta_f, delta_x):
        eta = beta * eta

    return eta


def newtons_method(x_0, func, df, d2f, search, conv_tol=10e-6, params=[10e-4, .5]):
    """Performs Newton's method on func with initial conditions of x_0

    Args:
        x_0: A two element numpy array holding the initial points
        func: A method which takes in a two element numpy array and returns a float.  This
        method should represent the function that we want to take the gradient of
        df:   A method which takes in a two element numpy array and returns a two element
        numpy array of the gradient of func
        d2f: A method which takes in a two element numpy array and returns a two dimensional
        numpy array of the hessian of func
        search: Search method for the optimal step size.  This method takes four parameters
        all four are two element numpy arrays the first is the current x the second is our change in x
        the third is our change in the function, and the fourth is an alpha and beta parameter for line search
        conv_tol: The convergense tolerance a float
        params: Optional two element numpy array holding search parameters

    Returns:
        x: two element numpy array which is the optimal x point.
        num_iter: integer The number of iterations until convergence
        x_vals: A list of two element numpy arrays filled with each step timepoint
    """
    x = x_0
    num_iter = 0
    x_vals = [x_0]
    l = sys.maxint
    while l / 2 > conv_tol:
        delta_x = - np.dot(np.linalg.inv(d2f(x)), df(x))
        delta_x = np.array([delta_x[0, 0]], delta_x[0, 1])
        l = np.dot(np.dot(np.transpose(df(x)), np.linalg.inv(d2f(x))), df(x))
        eta = search(x, delta_x, -delta_x, func, params)
        x = x + eta * delta_x
        num_iter += 1
        x_vals.append(x)

    return x, num_iter, x_vals


def plot_contours(func, dim, x_vals=None, t=""):
    """This method plots the contours of a function fun in the range of dim
    If x_vals are given then it plots these as lines and points on the graph
    to help visualize gradient descent

    Args:
        func: A function for which we want to display the contours
        dim: An array with four elements [x_min, x_max, y_min, y_max] to display
        contours in.
        x_vals: An array of numpy arrays with the points from each step of gradient descent
    Returns:
        Nothing but displays a plot of the contours
    """
    # Create meshed grid
    x = linspace(dim[0], dim[1], 300)
    y = linspace(dim[2], dim[3], 300)
    X, Y = meshgrid(x, y)
    Z = np.ones(X.shape)
    rows, cols = X.shape

    # Calculate func at each point in the meshed grid
    for i in range(0, rows):
        for j in range(0, cols):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # Find the contours
    contours = contour(X, Y, Z, logspace(-2, 3, 20))

    # Plot out each step of gradient descent on contours of func
    if(x_vals != None):
        points = np.matrix(x_vals)
        plot(points[:, 0], points[:, 1], '-or', ms=5)

    title(t)
    show()


def main():
    print "Perform Gradient Descent on Quadratic function"
    x_0 = np.array([10, 1])
    x_opt, num_iter, x_vals = grad_descent(x_0, f1, gradient_f1, backtracking_line_search, params=[.5, .99])
    print x_opt
    print num_iter
    print f1(x_opt)
    dim = [-10, 10, -4, 4]
    plot_contours(f1, dim, x_vals=x_vals, t="Gradient Descent with Backtracking Line Search")

    print("Examine Finite Difference of Rosenbrock Function")
    epsilon = 10e-4
    x = np.array([2.0, 2.0])
    grad = gradient_f(x)
    finite_diff = finite_difference(x, f, epsilon)
    print(grad - finite_diff)

    print("Perform backtracking line search")
    beta = .5
    alpha = 10e-4
    conv_tol = 10e-5
    num_iter = 0
    x_0 = np.array([1.2, 1.2])
    x_opt, num_iter, x_vals = grad_descent(x_0, f, gradient_f, backtracking_line_search)
    print x_opt
    print num_iter
    print f(x_opt)

    # plot the contours
    #dim = [-5, 5, -4, 3]
    #dim = [0, 2, 0, 2]
    #dim = [.5, 1.5, .5, 1.5]
    dim = [.9, 1.3, .9, 1.3]
    plot_contours(f, dim, x_vals=x_vals, t="Rosenbrock Function GD x_0 = [1.2,1.2]")

    # Redo with different initial condition
    x_0 = np.array([-1.2, 1])
    x_opt, num_iter, x_vals = grad_descent(x_0, f, gradient_f, backtracking_line_search)
    print x_opt
    print num_iter
    print f(x_opt)
    dim = [-1.3, 1.1, -1, 2]
    plot_contours(f, dim, x_vals=x_vals)

    print("Perform Newton's Method")
    x_0 = np.array([1.2, 1.2])
    x_opt, num_iter, x_vals = newtons_method(x_0, f, gradient_f, hessian_f, backtracking_line_search)
    print x_opt
    print num_iter
    print f(x_opt)
    dim = [.9, 1.3, .9, 1.3]
    plot_contours(f, dim, x_vals=x_vals, t="Rosenbrock Function Newton's Method x_0=[1.2,1.2]")

    # Redo with different initial condition
    x_0 = np.array([-1.2, 1])
    x_opt, num_iter, x_vals = newtons_method(x_0, f, gradient_f, hessian_f, backtracking_line_search)
    print x_opt
    print num_iter
    print f(x_opt)
    dim = [-1.3, 1.1, -1, 2]
    plot_contours(f, dim, x_vals=x_vals)


if __name__ == "__main__":
    main()
