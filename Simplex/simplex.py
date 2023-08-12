"""Volume 2: Simplex

<Name>
<Date>
<Class>
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Check for feasibility at the origin
        if np.min(b) < 0:
            raise ValueError("Origin not feasible")
        
        # Otherwise store each of the inputs as the dictionary
        else:
            self.m = len(b)
            self.n = len(A[0])
            self.dictionary = self._generatedictionary(c, A, b)


    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # Initialize the dictionary and fille in the first row and column
        dictionary = np.zeros((self.m+1, self.n+self.m+1))
        dictionary[1:,0] = b
        dictionary[0,1:self.n+1] = c

        # Fill in the rest of the dictionary with -A and the negative identity matrix
        dictionary[1:,1:self.n+1] = -np.array(A)
        dictionary[1:,self.n+1:] = -np.eye(self.m)
        return dictionary
    

    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # Loop through the first row and return the first index that is negative
        for i in range(1, len(self.dictionary[0])):
            if self.dictionary[0][i] < 0:
                return i


    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # Initialize the ratio list and get the column to pivot on
        ratiolist = [] 
        col = index

        # Loop through the rows and get the denominator
        for i in range(1, len(self.dictionary)):
            denom = self.dictionary[i][col]

            # If the denominator is zero, add infinity to the list and continue
            if denom == 0:
                ratiolist.append(np.inf)
                continue

            # calculate the ratio and test if it is postivie
            ratio = -self.dictionary[i][0] / denom
            if ratio > 0:
                ratiolist.append(ratio)

            # If the ratio is not positive, add infinity to the list
            else:
                ratiolist.append(np.inf)
        
        # Raise a value error if the list is all positive
        if np.min(ratiolist) == np.inf:
            raise ValueError("Unbounded problem")
        
        # Return the index of the minimum ratio
        return np.argmin(ratiolist) + 1
        
        
    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        # Get the pivot column and row
        col = self._pivot_col()
        row = self._pivot_row(col)

        # Get the pivot value
        k = self.dictionary[row][col]
        self.dictionary[row] /= -k
        
        # Zero out the other values in the column
        for i in range(len(self.dictionary)):
            if i != row:
                self.dictionary[i] += self.dictionary[row] * self.dictionary[i][col]
        

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        
        # Loop until the dictionary is optimal
        while np.min(self.dictionary[0,1:]) < 0:
            self.pivot()
        
        # Get the variables from the dictionary
        min = self.dictionary[0][0]
        dependent = {}
        independent = {}

        # Create the dependent variable dictionary
        for i in range(1, self.m + 1):
            # find the index of the pivot value and add it to the dictionary
            index = np.argmin(self.dictionary[:,i])
            dependent[i-1] = self.dictionary[index][0]

        # Create the independent variable dictionary, then return the value and dictionaries
        for i in range(self.n + 1, self.n+self.m):
            independent[i] = 0
        return min, dependent, independent


# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # Load the data and solve the problem
    mix = np.load(filename)

    # Get the data from the dictionary
    A = mix['A']
    p = mix['p']
    m = mix['m']
    d = mix['d']

    # Construct the new matrix and b using all constraints
    b = np.concatenate((m, d))
    k = len(d)
    A = np.concatenate((A, np.eye(k)), axis=0)
    
    # Create the solver and solve the problem
    solver = SimplexSolver(-p, A, b)
    min, dependent, independent = solver.solve()
    return [dependent[i] for i in range(len(p))]