# Newton's Method Project Description

Newton’s method, the classical method for finding the zeros of a function, is one of the most important algorithms of all time. In this project I implement Newton’s method in arbitrary dimensions and use it to solve a few interesting problems. I also explore in some detail the convergence (or lack of convergence) of the method under various circumstances.

## Mathematical Background and Overview of Functions

Newton's method, also known as the Newton-Raphson method, is an iterative method that starts with an initial guess for a root of a given function and refines the guess iteratively. Mathematically, the iterative formula can be represented as:
\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \]
where \( f'(x_n) \) is the derivative of \( f \) at the point \( x_n \).

1. **Newton's Method**
    - `newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.)`: This function applies Newton's method to approximate a zero of the function `f`. The parameter `x0` is the initial guess, `Df` is the derivative of the function, and `alpha` can be used to control the step size.

2. **Financial Mathematics with Newton's Method**
    - `prob2(N1, N2, P1, P2)`: Financial equations often involve interest rates and present values. This function uses Newton's method to find the constant `r` (interest rate) that equates two present values given certain conditions.

3. **Optimal Alpha Search**
    - `optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15)`: In some cases, adjusting the step size (`alpha`) can lead to faster convergence in Newton's method. This function seeks the optimal `alpha` value in the interval (0,1] that leads to the fastest convergence for the given function and initial point.

4. **Initial Point for Convergence to Specific Zeros**
    - `prob6()`: Certain initial points when plugged into Newton's method can lead the method to converge to specific roots. This function aims to find such initial points.

5. **Plotting Basins of Attraction on Complex Plane**
    - `plot_basins(f, Df, zeros, domain, res=1000, iters=15)`: The basins of attraction represent regions in the complex plane where all initial guesses converge to the same root. This function visualizes these basins for a given function.

## How to Use

1. Import the necessary functions from this module.
2. Define or provide the appropriate function, its derivative, initial guess, and other parameters for the function you intend to use.
3. Call the function and interpret the results.

## Dependencies

- numpy
- sympy
- matplotlib
- scipy

## Testing Code

To run a test, uncomment the desired test function at the bottom of the script. For instance, to test Newton's method on the function \( f(x) = x^3 - 1 \):

```python
testNewton1()