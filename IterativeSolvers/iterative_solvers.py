# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
        as_sparse: If True, an equivalent sparse CSR matrix is returned.
    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix."""
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = sparse.dok_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr()          # convert to row format for the next step
    for i in range(n):
        A[i,i] = abs(B[i]).sum() + 1
    return A.tocsr() if as_sparse else A.toarray()


# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot = False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # Get the diagonal of and get initial guess of x. Initialize the error list.
    d = np.diag(A)
    x0 = np.zeros_like(b)
    error = [la.norm(b, np.inf)]
    
    # Loop through the iterations and update each term
    for i in range(maxiter):
        x = x0 + (b - A @ x0) / d

        # Check for convergence and update x0 using the infinity norm
        if la.norm(x - x0, np.inf) < tol:
            break
        x0 = x

        # Append the error to the list if plot is True
        if plot:
            error.append(la.norm(A@x - b, np.inf))

    # Plot the error if plot is True
    if plot:
        plt.semilogy(error)
        plt.title("Convergence of Jacobi Method")

        # Label the axes and show the plot
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()

    # Return the solution
    return x


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize the first guess and get the diagonal
    x0 = np.zeros_like(b)
    d = np.diag(A)
    x = x0.copy()
    error = [la.norm(b, np.inf)]
    
    # Loop through the iterations throughout maxiter
    for i in range(maxiter):
        # Update each term in x
        for j in range(len(x)):
            x[j] = x[j] + (b[j] - A[j]@ x) / d[j]

        # Check for convergence and update x0 using the infinity norm
        if la.norm(x - x0, np.inf) < tol:
            break

        # Update the x0 and append the error to the list if plot is True
        x0 = x.copy()
        if plot:
            error.append(la.norm(A@x - b, np.inf))

    # Plot the error if plot is True
    if plot:
        plt.semilogy(error)
        plt.title("Convergence of Gauss-Seidel Method")

        # Label the axes and show the plot
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.show()

    # return the solution
    return x


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize the first guess and get the diagonal
    x0 = np.zeros_like(b)
    d = A.diagonal()
    x = x0.copy()
    
    # Loop through the iterations throughout maxiter
    for i in range(maxiter):
        # Update each term in x
        for j in range(len(x)):

            # get rowstart and rowend and get Aix
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]

            # update the jth element of x
            x[j] = x[j] + (b[j] - Aix) / d[j]

        # Check for convergence and update x0 using the infinity norm
        if la.norm(x - x0, np.inf) < tol:
            break
        x0 = x.copy()

    # return the solution
    return x


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # Initialize the first guess and get the diagonal
    x0 = np.zeros_like(b)
    d = A.diagonal()
    x = x0.copy()
    conv = False

    # Loop through the iterations throughout maxiter
    for i in range(maxiter):
        # Update each term in x
        for j in range(len(x)):

            # get rowstart and rowend and get Aix
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]

            # update the jth element of x
            x[j] = x[j] + omega * (b[j] - Aix) / d[j]

        # Check for convergence and update x0 using the infinity norm
        if la.norm(x - x0, np.inf) < tol:
            conv = True
            break
        x0 = x.copy()

    # return the solution
    return x, conv, i


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    # Handle cases where n is 1 or less
    if n <= 0:
        raise ValueError("n must be greater than 0")
    elif n == 1:
        A = sparse.csr_matrix(np.array([[-4]]))

    # Handle cases where n is greater than 1 to generate A 
    else:
        B = sparse.diags([1,-4,1], [-1,0,1], shape = (n,n))
        A = sparse.block_diag([B]*(n))
        A.setdiag(1,n)
        A.setdiag(1,-n)

    # construct our b vector
    builder = np.zeros(n)
    builder[0], builder[-1] = -100, -100
    b = np.tile(builder,n)

    # solve the system
    u, conv, iters = sor(A.tocsr(), b, omega, tol, maxiter)

    # if plot is true, then plot it using plt.colormesh as a heat map.
    if plot:
        plt.pcolormesh(u.reshape(n,n), cmap = "coolwarm")
        plt.colorbar()
        
        # Give it a title
        plt.title("Heat Map of u with n = " + str(n) + " and omega = " + str(omega))
        plt.show()

    # return the solution, convergence, and iterations
    return u, conv, iters


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    # Initialize the omega values and the number of iterations
    omegas = np.round(20 * np.linspace(1,1.95,20)) / 20
    n = 20
    maxiter = 1000
    
    # Initialize the number of iterations and tolerance
    tol = 1e-2
    iters = np.zeros_like(omegas)

    # Loop through the different omegas and get the iters
    for i,w in enumerate(omegas):
        u, conv, iteration = hot_plate(n, w, tol, maxiter, plot=False)
        iters[i] = iteration

    # Plot the number of iterations as a function of omega
    plt.plot(omegas, iters)
    plt.xlabel("Omega")
    plt.ylabel("Iterations")

    # Give it a title and show it
    plt.title("Number of Iterations as a Function of Omega")
    plt.show()

    # return the w that minimizes the number of iterations
    return omegas[np.argmin(iters)]