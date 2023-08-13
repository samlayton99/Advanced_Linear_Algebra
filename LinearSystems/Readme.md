# Linear Systems Project Description

The fundamental problem of linear algebra is solving the linear system Ax = b, given that a solution exists. There are many approaches to solving this problem, each with different pros and cons. In this project I implement the LU decomposition and use it to solve square linear systems. I also use SciPy, together with its libraries for linear algebra and working with sparse matrices.


## Mathematical Background and Overview of Functions

1. **Reduced Row Echelon Form (RREF)**
    - `ref(A)`: This function takes in a square matrix `A` and reduces it to its Reduced Row Echelon Form (RREF). It assumes that `A` is invertible, and that a zero will never appear on the main diagonal.

2. **LU Decomposition**
    - `lu(A)`: This function computes the LU decomposition of a square matrix `A`. It returns two matrices, `L` and `U`, representing the lower-triangular and upper-triangular parts of the decomposition, respectively.

3. **Solve Linear System Using LU Decomposition**
    - `solve(A, b)`: Uses the LU decomposition and back substitution to solve the linear system `Ax = b`. This function assumes that no row swaps are required for the decomposition.

4. **Comparison of Different Solving Methods**
    - `prob4()`: This function times different `scipy.linalg` functions for solving square linear systems and plots the results. The comparison includes inverting matrix A, using `la.solve()`, and using the LU decomposition for solving.

5. **Sparse Matrix Generation**
    - `prob5(n)`: Given an integer `n`, this function generates a `(n**2,n**2)` sparse matrix `A` based on a specific configuration involving an identity matrix `I` and matrix `B`.

6. **Comparing Regular and Sparse Linear System Solvers**
    - `prob6()`: This function times how long it takes to solve a system `Ax = b` using a regular solver and a sparse solver. It then plots the system size versus the execution times to compare the two methods' performances.

## How to Use

1. Import the necessary functions from this module.
2. Create or provide the appropriate matrix (or matrices) and vector inputs for the function you want to use.
3. Call the function and interpret the results.

## Dependencies

- numpy
- scipy
- matplotlib

## Examples

To reduce a matrix `A` to its RREF:
```python
A = [[1,2,3],[0,1,4],[5,6,0]]
reduced_A = ref(A)
print(reduced_A)
