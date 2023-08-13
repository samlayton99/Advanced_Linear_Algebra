# Numerical Conditioning and Stability in Python
The condition number of a function measures how sensitive that function is to changes in the input. On the other hand, the stability of an algorithm measures how accurately that algorithm computes the value of a function from exact input. Both of these concepts are important for answering the crucial question, “is my computer telling the truth?” In this project I examine the conditioning of common linear algebra problems, including computing polynomial roots and matrix eigenvalues. I also present an example to demonstrate how two different algorithms for the same problem may not have the same level of stability. This repository contains Python code that demonstrates concepts related to numerical conditioning and stability. The code utilizes libraries such as NumPy, SciPy, and SymPy to explore and illustrate these mathematical principles.

This repository contains Python code that explores fundamental concepts of linear algebra related to matrix condition, polynomial root stability, eigenvalue problems, polynomial approximation, and integral computation. The code is organized into several functions, each addressing a specific problem within the context of these concepts.

## Overview of Functions

1. `matrix_cond(A)`: Calculates the condition number of a matrix `A` using the 2-norm, indicating the sensitivity of the matrix to input changes.

2. `prob2()`: Demonstrates the impact of coefficient perturbations on polynomial root stability, generating plots to visualize the effects.

3. `eig_cond(A)`: Approximates the condition numbers of the eigenvalue problem for a given square matrix `A`, evaluating both the absolute and relative condition numbers.

4. `prob4(domain=[-100, 100, -100, 100], res=50)`: Visualizes the relative condition numbers of the eigenvalue problem over a defined grid, offering insights into the stability of eigenvalues.

5. `prob5(n)`: Approximates data using a least squares polynomial of degree `n`, compares solutions obtained via normal equations and QR decomposition, and visualizes the results.

6. `prob6()`: Compares SymPy-based integral computation with subfactorial formula-based estimation, analyzing relative forward error across varying values of `n`.

## Usage

To use these functions, ensure you have the following libraries installed:

- NumPy
- SymPy
- SciPy
- Matplotlib

Run each function or use the provided testing functions to validate their behavior. Each function is documented with a brief description, input parameters, and expected outputs.

## Testing

The code includes testing functions (`test1()`, `test2()`, `test3()`, `test4()`) that allow you to validate the correctness and behavior of the individual functions. Running these tests can help ensure the reliability of the provided code.

Please feel free to explore the code, modify the parameters, and adapt it to your needs. The project aims to provide insights into fundamental concepts of linear algebra and their implications for stability and accuracy in computations.
