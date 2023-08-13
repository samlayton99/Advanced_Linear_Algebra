# Iterative Solvers
In tackling real-world problems where systems take the form Ax = b with tens of thousands of parameters, the traditional approaches like Gaussian elimination or matrix factorizations become infeasible due to the potential for trillions of FLOPs. To address this, I've implemented three widely used iterative techniques—Jacobi, Gauss-Seidel, and Successive Over-Relaxation—aimed at approximating solutions for large systems. These methods prove invaluable in handling the intricacies of extensive equation systems. This repository offers Python code that embodies these iterative solvers, encompassing the Jacobi, Gauss-Seidel, and Successive Over-Relaxation (SOR) Methods. The codebase also includes functionalities for generating diagonally dominant matrices and solving linear equation systems via these iterative approaches. Moreover I demonstrate visualization of these solvers using the heat equation, utilizing the SOR Method for illustrative purposes.

## Overview of Functions

1. `diag_dom(n, num_entries=None, as_sparse=False)`: Generates a strictly diagonally dominant matrix of size n x n, with options to return the matrix as a dense or sparse format.

2. `jacobi(A, b, tol=1e-8, maxiter=100, plot=False)`: Calculates the solution to the system Ax = b using the Jacobi Method, with options to specify convergence tolerance and maximum iterations, and plot the convergence rate.

3. `gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False)`: Calculates the solution to the system Ax = b using the Gauss-Seidel Method, with similar options as the Jacobi Method.

4. `gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100)`: Calculates the solution to a sparse system Ax = b using the Gauss-Seidel Method.

5. `sor(A, b, omega, tol=1e-8, maxiter=100)`: Calculates the solution to the system Ax = b using the Successive Over-Relaxation (SOR) Method, with an additional parameter for the relaxation factor.

6. `hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False)`: Generates and solves a system representing a heat equation using the SOR Method, with options to visualize the solution as a heatmap.

7. `prob7()`: Runs the `hot_plate()` function for different relaxation factors and plots the number of iterations as a function of omega.

## Usage

To use these functions, ensure you have the following libraries installed:

- NumPy
- SciPy
- Matplotlib

You can run each function individually or use them collectively to solve linear systems of equations and visualize solutions. Each function is documented with a brief description, input parameters, and expected outputs.

## Testing

The code includes a testing function `prob7()` that demonstrates the behavior of the `hot_plate()` function with varying relaxation factors. Running this test can help you understand the convergence behavior of the Successive Over-Relaxation Method.

Feel free to explore the code, modify the parameters, and adapt it to your needs. The project aims to provide insights into iterative methods for solving linear systems of equations and their applications in computational mathematics.