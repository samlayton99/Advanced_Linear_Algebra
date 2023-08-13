# Least Squares Eigenvalues Project Description

Because of its numerical stability and convenient structure, the QR decomposition is the basis of many important and practical algorithms. In this project I introduce linear least squares problems, tools in Python for computing least squares solutions, and two fundamental algorithms for computing eigenvalue. The QR decomposition makes solving several of these problems quick and numerically stable.

## Overview of Functions and Mathematical Background

- **least_squares**: Computes the least squares solutions to the system `Ax = b` using QR decomposition. The method uses QR decomposition to determine the least squares solution to a system of equations by factorizing matrices into an orthogonal matrix `Q` and an upper triangular matrix `R`.

- **line_fit** and **polynomial_fit**: Computes and plots the least squares line for the housing price index versus years and fits and plots least squares polynomials of varying degrees for the housing price index data, respectively. These functions apply the principles of least squares to fit lines and polynomials to data.

- **ellipse_fit**: Computes and plots the best fitting ellipse for a given dataset. The method leverages least squares to fit an ellipse defined as `ax^2 + bx + cxy + dy + ey^2 = 1` to a set of data points by adjusting the coefficients a, b, c, d, and e.

- **power_method**: Computes the dominant eigenvalue of a matrix and its corresponding eigenvector. The method determines the dominant eigenvalue (largest in magnitude) and its associated eigenvector using an iterative approach.

- **qr_algorithm**: Computes the eigenvalues of a matrix using the QR algorithm. The algorithm finds the eigenvalues by factorizing a matrix into its QR decomposition, then recomposing them in a reverse order. Repeated application converges the matrix to a triangular form, revealing its eigenvalues.

## Usage

To use these functions, ensure you have the following libraries installed:

- `scipy`
- `numpy`
- `matplotlib`

For further information on the mathematical background of these functions and their applications, please refer to standard numerical analysis textbooks or computational mathematics resources. 