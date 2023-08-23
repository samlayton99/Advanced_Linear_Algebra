# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Sam Layton>
<Volume 1 Section 003>
<10/16/22>
"""


import scipy.linalg as la
import cmath
import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    # Get the QR decomposition of A.
    Q,R = la.qr(A, mode='economic')

    # Solve the system Rx = Q^T b for x.
    return la.solve_triangular(R, Q.T@b)


# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # Load the data and separate the columns.
    data = np.load("input_files/housing.npy")
    v1, b = data[:,0], data[:,1]

    # Create the matrix A and find the least squares solution.
    A = np.column_stack((v1, np.ones(len(v1))))
    x = least_squares(A, b)

    # Plot the data and the least squares line and give a legend.
    plt.plot(v1, b, 'o')
    plt.plot(v1, A@x, '--')
    plt.legend(["Actuals", "Least Squares Line"])

    # Label the axes and show the plot.
    plt.title("Least Squares Approximation of Housing Data")
    plt.xlabel("Years since 2000")
    plt.ylabel("HPI")
    plt.show()

    
# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    # Initialize the degree, column vectors, and load the data.
    degree = [3, 6, 9, 12]
    data = np.load("input_files/housing.npy")
    v1, b = data[:,0], data[:,1]

    # Create the matrix A and make a smooth domain using np.linspace for plotting.
    A = []
    v1_smooth = np.linspace(v1[0], v1[-1], 100)
    Legend = ["Actuals"]

    # Loop through the degrees and create the matrix A.
    for i in range(len(degree)):
        # Create the vander matrix for each degree.
        matrix = np.vander(v1, degree[i] + 1)

        # Find the least squares solution and make a polynomial
        coeff = least_squares(matrix, b)
        poly = np.poly1d(coeff)

        # Run the polynomial through the data and append to A
        fitted = poly(v1_smooth)
        A.append(fitted)
        Legend = Legend + ["Degree " + str(degree[i])]

    # Convert A to a numpy array and plot the data and the least squares lines.
    A = np.array(A)
    
    # Plot all four subplots in one figure.
    for i in range(len(degree)):
        plt.subplot(2,2,i+1)
        plt.plot(v1, b, 'o')
        plt.plot(v1_smooth, A[i], '--')

        # Label the axes and give a legend.
        plt.legend(["Actuals" ,"Degree " + str(degree[i])])
        plt.xlabel("Years since 2000")
        plt.ylabel("HPI")

    # Show the plot neatly
    plt.tight_layout()
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")
    

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    # Load the data and separate the columns.
    data = np.load("input_files/ellipse.npy")
    x, y = data[:,0], data[:,1]

    # Create the matrix A and find the least squares solution.
    A = np.column_stack((x**2, x, x*y, y, y**2))
    a,b,c,d,e = least_squares(A, np.ones(len(x)))

    # Plot the data and the ellipse and give a legend.
    plt.plot(x, y, 'o')
    plot_ellipse(a,b,c,d,e)
    plt.legend(["Actuals", "Ellipse"])
    
    # Label the axes and show the plot.
    plt.title("Least Squares Approximation of Ellipse")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    # Get the matrix dimensions and initialize the eigenvector.
    m,n = np.shape(A)
    x = np.random.rand(n)

    # normalize the eigenvector and initialize i.
    x = x / la.norm(x)
    i = 0

    # Loop through until the tolerance is met or the max iterations are reached.
    while i < N:
        # Calculate the new eigenvector.
        x_new = A@x

        # If the new eigenvector is within tolerance, break the loop.
        if la.norm(x_new - x) < tol:
            break

        # Normalize the new eigenvector and increment i.
        x = x_new
        x = x / la.norm(x)
        i += 1
    
    # Return the eigenvalue and eigenvector.
    return x.T@A@x, x
        

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m, n = np.shape(A)
    S = la.hessenberg(A)

    # Loop through the number of iterations.
    for i in range(N):
        # Find the QR decomposition of S and update S.
        Q, R = la.qr(S)
        S = R@Q
    # Initialize the eigenvalues and loop through the diagonal of S.
    eigenvalues = []
    i = 0

    # Loop through the diagonal of S.
    while i < n:
        # Check if the diagonal block is 1x1 or 2x2.
        if i == n-1 or abs(S[i+1, i]) < tol:
            eigenvalues.append(S[i,i])

        # If the diagonal block is 2x2, find the eigenvalues and append them.
        else: 
            # get the values of the diagonal block. and implement the quadratic formula.
            a, b, c, d = S[i,i], S[i,i+1], S[i+1,i], S[i+1,i+1]
            p2, p3 = -(a+d), a*d - b*c

            # Find the eigenvalues using the quadratic formula.
            eig1 = (-p2 + cmath.sqrt(p2**2 - 4*p3))/2
            eig2 = (-p2 - cmath.sqrt(p2**2 - 4*p3))/2

            # Append the eigenvalues and increment i.
            eigenvalues.append(eig1)
            eigenvalues.append(eig2)
            i += 1
        i += 1
    
    # Return the eigenvalues.
    return eigenvalues