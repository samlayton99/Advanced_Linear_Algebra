# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
import matplotlib.pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    # calculate the singular values of A and check if any are zero and return infinity if so
    sigmas = la.svdvals(A)
    if np.min(sigmas) < 1e-14:
        return np.inf

    # if not, return the ratio of the largest to smallest singular value
    else:
        return np.max(sigmas)/np.min(sigmas)


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    # Get the roots of the original Wilkinson polynomial and initialize the condition numbers list.
    w_roots = np.arange(1, 21)
    cond_nums = []

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    # Repeat the experiment 100 times and get 20 normal random numbers for each.
    for i in range(100):
        r = np.random.normal(1, 1e-10, 21)

        # Perturb the coefficients of the Wilkinson polynomial and get the roots.
        w_pert = w_coeffs * r
        w_pert_roots = np.roots(w_pert)
        
        # Plot the roots of the perturbed polynomial and get the abosolute and relative condition numbers.
        plt.plot(w_pert_roots.real, w_pert_roots.imag, '.', color='blue', alpha = .5, markersize = 1)
        cond_nums.append(la.norm(w_pert_roots - w_roots, np.inf) / la.norm(r, np.inf))

    # Give it a title, plot the original roots, give it a legend, and show the plot.
    plt.title("Roots of the Perturbed Wilkinson Polynomial")
    plt.plot(w_roots.real, w_roots.imag, 'x', color='red', markersize = 5, label = "Original Roots")
    plt.legend()
    plt.show()

    # Get the average absolute and relative condition numbers and return them
    absolute = np.mean(cond_nums)
    relative = absolute * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)
    return absolute, relative
    

# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # Get the eigenvalues of A
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    
    # Get the eigenvalues of A+H and reorder them to be as close to the original eigenvalues as possible.
    eigvals = la.eigvals(A)
    pert_eigvals = la.eigvals(A+H)
    pert_eigvals = reorder_eigvals(eigvals, pert_eigvals)

    # Get the absolute and relative condition numbers and return them.
    absolute = la.norm(eigvals - pert_eigvals,2) / la.norm(H, 2)
    relative = absolute * la.norm(A, 2) / la.norm(eigvals, 2)

    # Return the absolute and relative condition numbers.
    return absolute, relative


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # Make a mesh grid over the domain and have the resolution be res.
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[2], domain[3], res)
    X, Y = np.meshgrid(x, y)

    # Make the lambda function
    matrix = lambda x, y: np.array([[1, x], [y, 1]])

    # Get the absolute and relative condition numbers for each point in the mesh grid.
    relative = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            relative[i,j] = eig_cond(matrix(x[i],y[j]))[1]

    # Plot the relative condition numbers
    plt.pcolormesh(X, Y, relative, cmap='gray_r', )
    plt.colorbar()
    plt.title("Relative Condition Number of the Eigenvalue Problem")

    # Make the plot look nice and show it
    plt.tight_layout()
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # Load the data and build the Vandermonde matrix
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)

    # Solve the least squares problem using the inverse and QR decomposition
    inverse = la.inv(A.T @ A) @ A.T @ yk
    Q, R = la.qr(A, mode='economic')
    qrdecomp = la.solve_triangular(R, Q.T @ yk)

    # Get the mean squared error for each solution

    # get the polynomials for plotting
    inverse1 = np.poly1d(inverse)
    qrdecomp1 = np.poly1d(qrdecomp)

    # Plot the polynomials of each and label them
    plt.plot(xk, yk, 'o', label="Data")
    plt.plot(xk, inverse1(xk), label="Inverse")
    plt.plot(xk, qrdecomp1(xk), label="QR Decomposition")

    # Give it a legend and title and show it
    plt.legend()
    plt.title("Polynomial Approximation of Stability Data")
    plt.show()

    # Return the mean squared error for each solution
    return la.norm(A @ inverse1 - yk, 2), la.norm(A @ qrdecomp1 - yk, 2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    # Define the symbols and integrate for all n, make a list of the values
    x, n = sy.symbols('x k')
    plotlist = []
    for n in range(5,55,5):
        exp = sy.integrate((x**int(n)) * sy.exp(x-1), (x, 0, 1))

        # Do our step 2 and get the forward relative error
        val = float(((-1) ** n) * (sy.subfactorial(n) - sy.factorial(n)/np.e))
        rel_error = abs((exp - val) / exp)

        # Append the relative error to the list and plot it
        plotlist.append(rel_error)
    plt.plot(np.linspace(5,55,10), plotlist, color='red')

    # Label the plot and axis
    plt.title("Relative Forward Error of Subfactorial Formula")
    plt.ylabel("Relative Forward Error")
    plt.xlabel("n")

    # Make the log scale and show it
    plt.yscale('log')
    plt.show()


# Test functions
def test1():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 3]])
    k = matrix_cond(A)
    print(k)
    print(np.linalg.cond(A, 2))
    t = la.qr(A)[0]
    print(np.linalg.cond(t))


def test2():
    prob2()


def test3():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 3]])
    k = eig_cond(A)
    print(k)
    
def test4():
    prob4()

