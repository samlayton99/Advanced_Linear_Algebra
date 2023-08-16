# Simplex Project Description

The Simplex Method is a straightforward algorithm for finding optimal solutions to optimization problems with linear constraints and cost functions. Because of its simplicity and applicability, this algorithm has been named one of the most important algorithms invented within the last 100 years. In this project I implement a standard Simplex solver for the primal problem.

## Mathematical Background and Overview of Functions

The Simplex algorithm, being a cornerstone for solving linear optimization problems, iteratively pivots through the vertices of the feasible region determined by constraints until it arrives at the optimal solution. The algorithm thrives on its ability to consistently refine the solution through a series of functions, each of which contributes to different stages of the problem-solving process.

### Feasibility at Origin:

This aspect ensures that the origin is a feasible solution by checking that all elements of `b` are non-negative.

### Dictionary Generation:

- `SimplexSolver.__init__(self, c, A, b)`: This function initializes the Simplex solver, ensures feasibility, and kickstarts the dictionary.
- `SimplexSolver._generatedictionary(self, c, A, b)`: Crafts the initial dictionary pivotal to the Simplex algorithm, leveraging the coefficient matrix `A`, the objective coefficients `c`, and the constraint vector `b`.

### Pivoting:

An essential step to improve the current solution:
- `SimplexSolver._pivot_col(self)`: Determines the column (or variable) that, when increased in value, will enhance the objective function.
- `SimplexSolver._pivot_row(self, index)`: Utilizes Bland's Rule to ascertain the tightest constraint, consequently deciding which variable will reduce to zero.
- `SimplexSolver.pivot(self)`: This function actualizes the pivot, resulting in updates to the dictionary.

### Problem Solution:

The culmination of the preceding functions, it solves the optimization problem:
- `SimplexSolver.solve(self)`: Employing the Simplex algorithm, this function eventually concludes the optimization problem, returning the minimum objective function value and values of both basic (dependent) and non-basic (independent) variables.

### Product Mix Problem:

A special application of the Simplex algorithm:
- `prob6(filename='productMix.npz')`: Tailored for the product mix problem, it uses data from an integrated .npz file. By intertwining both resource and demand constraints, this function deduces an optimal solution detailing the quantities for various products.

## How to Use

1. Start by importing the required functions from the module.
2. Clearly define or feed the function, its derivative, the initial guess, and other ancillary parameters relevant to the chosen function.
3. Invoke the function and subsequently interpret the outcomes.

## Dependencies

- numpy

## Errors and Exceptions

Two main exceptions can arise:

- If the problem fails to be feasible at the origin, a `ValueError` citing "Origin not feasible" is dispatched.
- In the event of an unbounded problem, a `ValueError` highlighting "Unbounded problem" is raised.

## Notes

Venturing into the module's depths, it's essential to grasp the Simplex algorithm's mathematical scaffolding. Such understanding illuminates the pivoting strategies at play, and the holistic optimization route.