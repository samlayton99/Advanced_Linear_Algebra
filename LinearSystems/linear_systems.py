# linear_systems.py
"""Volume 1: Linear Systems.
<Sam Layton>
<Vol1 Section 003>
<8/17/22>
"""
from email.errors import InvalidBase64CharactersDefect
import numpy as np
from scipy import linalg as la
from scipy import sparse
import time
import matplotlib.pyplot as plt
from scipy.sparse import linalg as spla


# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    # Convert to np array and get shape
    A = np.array(A, dtype = float)
    m = len(A)
    n = len(A[0])

    # Loop through the columns and rows, starting at 2nd row, and 1st column
    for j in range(n): #columns
        for i in range(j + 1,m): #rows

            # Calculate the scale, and then subtract the scaled row from the current row
            scale = A[i][j]/A[j][j]
            A[i] = A[i] - scale*A[j]
    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    # Get shape of A
    m = len(A)
    n = len(A[0])

    #Convert to an np array and initialize L and U
    A = np.array(A, dtype = float)
    L = np.eye(m)
    U = np.copy(A)

    # Loop through the columns and rows, then update the L and U matrices
    for j in range(n):
        for i in range(j + 1, m):

            # Let L be the scale, and then subtract the scaled row from the current row
            L[i][j] = U[i][j]/U[j][j]
            U[i][j:] = U[i][j:] - L[i][j]*U[j][j:]
    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    # Get L and U from LU decomposition
    L, U = lu(A)
    # Get shape of A
    m = len(A)
    n = len(A[0])

    # Initialize y and x
    y = np.zeros(m)
    x = np.zeros(m)

    # Solve for y
    for i in range(m):
        y[i] = b[i] - np.dot(L[i][:i], y[:i])
    
    # Solve for x
    for i in range(m-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i][i+1:], x[i+1:]))/U[i][i]
    return x


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    # Initialize arrays to store the times of each method
    inverse = []
    solve = []
    lu_factor_and_solve = []
    lu_solve_only = []

    # Loop through powers of 2
    k = 13
    for i in range(k):
        # Calculate n and generate A and b
        n = 2**i + 1
        A =np.random.random((n,n))
        b = np.random.random(n)

        # Time A inverse and left multiply
        start = time.time()
        x = la.inv(A)@b
        end = time.time()
        inverse.append(end - start)

        # Time la.solve
        start = time.time()
        x = la.solve(A,b)
        end = time.time()
        solve.append(end - start)

        # Time lu_factor and lu_solve
        start = time.time()
        lu, piv = la.lu_factor(A)
        x = la.lu_solve((lu, piv), b)
        end = time.time()
        lu_factor_and_solve.append(end - start)

        # Time lu_solve only
        start = time.time()
        x = la.lu_solve((lu, piv), b)
        end = time.time()
        lu_solve_only.append(end - start)
    
    # Generate X values for plotting
    X = [2**i + 1 for i in range(k)]
    plt.figure()

    # Plot the times
    plt.plot(X, inverse, label = "Inverse")
    plt.plot(X, solve, label = "Solve")
    plt.plot(X, lu_factor_and_solve, label = "LU Factor and Solve")
    plt.plot(X, lu_solve_only, label = "LU Solve Only")
    
    # Make log scale, add legend, and give title
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Time to Solve Linear Systems")

    # Label axes and show plot
    plt.xlabel("log(n)")
    plt.ylabel("log(Time)")
    plt.show()




# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """

    # Generate B of size n by n and join together in larger n^2 by n^2 matrix
    B = sparse.diags([1,-4,1], [-1,0,1], shape = (n,n))
    A = sparse.block_diag([B]*(n))

    # Add identity matrices to the left and right of A if n > 1
    if n > 1:
        A.setdiag(1,n)
        A.setdiag(1,-n)

    # Show the matrix in plt.spy and return A
    #plt.spy(A, markersize=1)
    #plt.show()
    return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    # Initialize arrays to store the times of each method
    sparsearray = []
    normalarray = []

    # Loop through powers of 2
    k = 55
    for i in range(1, k):
        # Generate A and B, put them in the right format, and generate b
        A = prob5(i).tocsr()
        B = A.toarray()
        b = np.random.random(i**2)

        # Time sparse solve
        start = time.time()
        spla.spsolve(A,b)
        end = time.time()
        sparsearray.append(end - start)
        
        # Time normal solve
        start = time.time()
        la.solve(B,b)
        end = time.time()
        normalarray.append(end - start)
    
    # Generate X values for plotting
    X = [i for i in range(1, k)]
    plt.figure()

    # Plot the times
    plt.plot(X, sparsearray, label = "Sparse Solve")
    plt.plot(X, normalarray, label = "Normal Solve")
    
    # Make log scale and add legend
    plt.legend()
    plt.yscale("log")
    plt.title("Sparse vs Normal Solve")

    # Label axes and show plot
    plt.xlabel("(n)")
    plt.ylabel("log(Time)")
    plt.show()
