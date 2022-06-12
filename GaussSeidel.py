import numpy as np
import matplotlib.pyplot as plt
import time


class GaussSeidel:
  def __init__(self, initialX, A, b, tolerance, maxIterations):
    self.x = initialX
    self.tolerance = tolerance
    self.A = A
    self.b = b
    self.maxIterations = maxIterations

  def gaussSeidel(self):
    # Diagonal matrix
    D = np.diag(self.A)

    # Lower diagonal
    L = np.tril(self.A, -1)

    # Upper diagonal
    U = np.triu(self.A, 1)
    iter = 0

    # Check whether there is a 0.0 in the diagonal of A
    if 0. in D:
      # If so, the jacobi method will not converge
      return np.array([np.NAN]), iter, None, None

    # (L+D)^-1
    inverseLD = np.linalg.inv(L+np.diag(D))

    # -(L+D)^-1
    B = np.asmatrix(-inverseLD) * np.asmatrix(U)

    # (L+D)^-1b
    C = np.asmatrix(inverseLD) * np.asmatrix(self.b)

    # x will initially be a vector with random values
    x = np.random.uniform(size=(np.size(self.b), 1))

    # The new vector x(newX) will be x0(the initial x guess) 
    newX = self.x

    # List of local errors
    # Calculate the infinite norm error value of the initial guess x(x0) and the random vector x
    localErrors = [np.linalg.norm(newX - x, np.inf)]

    # List of x solutions
    xSolutions = [self.x]

    # Start time
    startTime = time.time()
    
    while iter < self.maxIterations and localErrors[iter] > self.tolerance:
      x = newX
      iter += 1

      # Calculate a new x solution
      newX = np.asmatrix(B) * np.asmatrix(x) + np.asmatrix(C)

      # Add the local error to the list of local errors
      localErrors.append(np.linalg.norm(newX - x, np.inf))
      
      # Add the new solution x to the list of solutions
      xSolutions.append(newX)

    x = newX
    xSolutions.append(x)

    # End time
    endTime = time.time()
    print(f'Completed at {endTime - startTime} \n')

    return xSolutions, iter, localErrors, x
  
  # Plot the local erros
  def plotError(self, error):
    figure = plt.figure(1)
    plt.title("Error")
    plt.plot(error[1:])
    plt.grid()
    plt.show()