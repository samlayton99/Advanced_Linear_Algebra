# Advanced Linear Algebra

This repository presents a deep dive into linear algebra, illustrating a collection of pivotal concepts, algorithms, and applications. It presents the a wide range of the mathematical techniques and computational tools commonplace in various fields of study and industry. The presentation of these projects aims to provide a foundation for understanding, visualizing, and solving real-world problems using foundational principles of linear algebra.

## Modules:

- **Bayesian and Thompson Sampling**:
  - Focuses on Bayesian statistics and Thompson Sampling, using Bayes' theorem to update probability estimates from prior beliefs and new data.
  - Introduces Thompson Sampling algorithms for probabilistic action choices, utilizing Normal and Gamma distributions, and Markov Chain Monte Carlo (MCMC) methods for Bayesian inference.

- **Conditioning and Stability**:
  - Explores numerical conditioning and stability in linear algebra, emphasizing matrix condition, polynomial root stability, and eigenvalue problems using Python libraries.
  - Provides functions to calculate matrix condition numbers, visualize coefficient perturbations on polynomial roots, and assess different approximation techniques.

- **Iterative Solvers**:
  - Presents methods to solve large-scale linear systems efficiently, showcasing iterative techniques such as Jacobi, Gauss-Seidel, and Successive Over-Relaxation (SOR).
  - Includes functions to produce diagonally dominant matrices and visualize solution approximations, focusing on the application of the SOR Method on the heat equation.

- **Least Squares and Eigenvalues**:
  - Utilizes QR decomposition for numerical stability and structure in linear problem solving, demonstrating least squares solutions for data set fitting.
  - Introduces algorithms for computing eigenvalues: the "power_method" for dominant eigenvalue determination and the "qr_algorithm" using QR decomposition for matrix eigenvalue deduction.

- **Linear Systems**:
  - Offers tools to solve \(Ax = b\), majorly using the LU decomposition method, featuring functions like `ref(A)` for RREF and `lu(A)` for LU decomposition.
  - Contains comparative analysis functions, emphasizing performance variances between different techniques and the advantages of sparse solvers.

- **Newton’s Method**:
  - Explores Newton's iterative technique for finding function zeros, applying it through the `newton(f, x0, Df)` function with refinements for optimal convergence.
  - Uses the method in financial mathematics, particularly in determining interest rates, and offers visual insights into basins of attraction in the complex plane.

- **Simplex**:
  - Implements the Simplex Method for linear optimization problem-solving, iterating through vertices defined by linear constraints to find the optimal solution.
  - Features core functions for feasibility verification, initial dictionary generation, and solution pivoting, along with a specialized application for the product mix problem.

For more detailed information, navigate to the respective module folders. Ensure to integrate the necessary libraries as mentioned within each module for smooth execution.
