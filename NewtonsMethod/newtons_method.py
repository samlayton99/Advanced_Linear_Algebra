# newtons_method.py
"""Volume 1: Newton's Method.
<Sam Layton>
<001>
<2/1/23>
"""
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """

    # Set boolVal to False and check if x0 is a float or an array
    boolVal = False
    if np.isscalar(x0):

        # Loop through for the maximum number of iterations.
        for i in range(maxiter):
            # Compute the next approximation using Newton's method.
            x1 = x0 - alpha * f(x0) / Df(x0)

            # Check for convergence. Return the result if converged.
            if abs(x1 - x0) < tol:
                boolVal = True
                break

            # Update the current approximation. If finished, return the result.
            x0 = x1
        return x1, boolVal, i + 1


    else: 
    # Turn x0 into a numpy array
        x0 = np.array(x0)

        # Loop through for the maximum number of iterations and solve the system of equations
        for i in range(maxiter):
            x1 = x0 - alpha * la.solve(Df(x0), f(x0)).T

            # Check for convergence. Set boolVal to True if converged.
            if la.norm(x1 - x0, 2) < tol:
                boolVal = True
                break
            x0 = x1
        
        # Return the result, whether or not it converged, and the number of iterations
        return x1, boolVal, i + 1


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    # Define the r_0, sympy symbols, and function
    r_0 = 0.1
    r = sy.Symbol('r')
    expr = P1*((1+r)**N1 - 1) - P2*(1 - (1+r)**(-N2))

    # Define the function and its derivative
    f = sy.lambdify(r, expr, 'numpy')
    Df = sy.lambdify(r, sy.diff(expr, r), 'numpy')

    # Use Newton's method to find the root
    return newton(f, r_0, Df)[0]


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    # Create the alpha array and the array to store the number of iterations
    alpha = np.delete(np.linspace(0, 1, 1000),0)
    iters = np.zeros(len(alpha))

    # Loop through the alpha values and store the number of iterations
    for i in range(len(alpha)):
        iters[i] = newton(f, x0, Df, tol, maxiter, alpha[i])[2]

    # Plot the number of iterations vs. alpha and label the axes
    plt.plot(alpha, iters)
    plt.xlabel('alpha')
    plt.ylabel('iterations')

    # Give the graph a title, label the axes, and show the graph
    plt.title('Iterations vs. Alpha')
    plt.ylim(0, 16)
    plt.show()

    # Return the alpha value that results in the lowest number of iterations
    return alpha[np.argmin(iters)]


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    # Define the function and its derivative
    f = lambda x: np.array([5*x[0]*x[1] - x[0]*(1 + x[1]), -x[0]*x[1] + (1 - x[1])*(1 + x[1])])
    Df = lambda x: np.array([[5*x[1] - (1 + x[1]), 5*x[0] - x[0]], [-x[1], -x[0] - 2 * x[1]]])
    
    while True:
        # Pick a random point between [-1/4,0] and [0 ,1/4]
        x0 = np.array([np.random.uniform(-1/4, 0), np.random.uniform(0, 1/4)])

        # Run Newton's method with alpha = 1 and alpha = 0.55
        first = np.array(newton(f, x0, Df, alpha=1, maxiter=40)[0])
        second = np.array(newton(f, x0, Df, alpha=0.55, maxiter=100)[0])
        
        # Check if the first point is (0,1) or (0,-1) and the second point is (3.75, .25)
        if (np.allclose(first, np.array([0,1])) or np.allclose(first, np.array([0,-1]))) and np.allclose(second, np.array([3.75, .25])):
            return x0


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    # Construct a resxres grid over the domain using meshgrid
    x_real = np.linspace(domain[0],domain[1], res)
    x_imag = np.linspace(domain[2],domain[3], res)
    X_real, X_imag = np.meshgrid(x_real, x_imag)

    # Get the imiginary inputs
    X0 = X_real + 1j*X_imag
    
    # Run Newton's method on the grid
    for i in range(iters):
        X0 = X0 - f(X0)/Df(X0)

    # Initialize the array to store the basins and loop through the grid
    Y = np.zeros_like(X0)
    for i in range(res):
        for j in range(res):

            # Find the index of the closest zero to the current point
            Y[i,j] = np.argmin(np.abs(X0[i,j] - zeros))

    # Recast the array as integers
    Y = np.real(Y).astype(int)
    
    # Plot the basins
    plt.pcolormesh(X_real, X_imag, Y, cmap='brg')
    plt.show()
    



# Testing code
def testNewton1():
    """Test Newton's method on the function f(x) = x^3 - 1."""
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: 1./3 * np.power(np.abs(x), -2./3)
    x0 = .01
    alpha = .4
    tol = 1e-5
    maxiter = 15
    x, converged, iters = newton(f, x0, Df, tol, maxiter, alpha)
    print("x = {:.5f}".format(x))
    print("converged = {}".format(converged))
    print("iters = {}".format(iters))

def testNewtonArray():
    def f(x):
        x0 = x[0]
        x1 = x[1]
        return np.array([3*x0**2 - x1 - 1, x0 + 3*x1**2 - 1])

    def Df(x):
        x0 = x[0]
        x1 = x[1]
        return np.array([[6*x0, -1], [1, 6*x1]])
    
    x0 = np.array([1.5, 1.5])
    alpha = 0.5
    tol = 1e-5
    maxiter = 30
    x, converged, iters = newton(f, x0, Df, tol, maxiter, alpha)
    print("x = {}".format(x))
    print("converged = {}".format(converged))
    print("iters = {}".format(iters))

def testProb2():
    P1 = 30
    P2 = 20
    N1 = 2000
    N2 = 8000
    r = prob2(P1, P2, N1, N2)
    print("r = {:.5f}".format(r))

def testOptimalAlpha():
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: 1./3 * np.power(np.abs(x), -2./3)
    x0 = 1
    alpha = optimal_alpha(f, x0, Df)
    print("alpha = {:.5f}".format(alpha))

def testProb6():
    x0 = prob6()
    print("x0 = {}".format(x0))

def justforfun(n):
    plot = np.zeros((n,2))
    for i in range(n):
        x, y = prob6()
        plot[i,0] = x
        plot[i,1] = y
        
    
    plt.plot(plot[:,0], plot[:,1], 'o')
    plt.show()

def testProb7(n=1000):
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    zeros = np.array([1, -1/2 + np.sqrt(3)/2*1j, -1/2 - np.sqrt(3)/2*1j])
    plot_basins(f, Df, zeros, domain = [-1.5, 1.5, -1.5, 1.5], res = n, iters = 15)



#testNewton1()
#testNewtonArray()
#testProb2()
#testOptimalAlpha()
#testProb6()
#testProb7()
#justforfun(100)