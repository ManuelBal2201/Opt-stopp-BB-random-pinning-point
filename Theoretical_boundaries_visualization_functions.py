# Libraries
import numpy as np
from scipy.stats import rv_discrete, rv_continuous, norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
import random

## Visualization
import matplotlib.pyplot as plt


# Boundary function for Brownian bridge
def optimal_stopping_Brownian_bridge(t, r):
  """

  Define the optimal stopping boundary given in (1) for for the particular case mu = r. REVISAR SI EL INDICE ES CORRECTO EN EL DEFINITIVO

  Parameters:
  - t (float): Time for which we want to know the optimal stopping boundary.
  - r (float): Pinning point.

  Return:
  - optimal_stopping_Brownian_bridge (function): Optimal stopping boundary function.

  """
  return r + 0.839924 * np.sqrt(1 - t)


# Theoretical optimal stopping boundary for a Normal distribution with standard deviation lower than 1
## Computation of error
def compute_error(boundary_new, boundary):
    """
    Compute relative error between iterations for convergence check.
    
    Parameters:
    - boundary_new (np.array): Boundary obtained in the current iteration.
    - boundary (np.array): Boundary obtained in the previous iteration.
    
    Returns:
    - error (np.array): Relative error between iterations.
    
    """
    numerator = np.linalg.norm(boundary_new - boundary)
    denominator = np.linalg.norm(boundary_new)
    return numerator / denominator if denominator != 0 else float('inf')


## f_t evaluated in the upper limits of the integrals
def f_t_sup(m, gamma, t, b_t, u, eval_mesh, b_tu, pers = "Eduardo"):
    """
    f_t evaluated in the upper limits of the integrals.
    
    Parameters:
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - t (float): Actual time t_n.
    - b_t (float): Actual value of the boundary.
    - u (np.array): Step between consecutives times of the eval_mesh.
    - eval_mesh (np.array): Points of the temporal grid that are the upper limits of the integrals.
    - b_tu (np.array): Boundary value for eval_mesh times.
    - pers (string): Perspective chosen. Eduardo or Abel
    
    Returns:
    - f_sup (np.array): f_t evaluated in the upper limits of the integrals.
    
    """
    c1 = (1 - gamma**2) # Constant 1
    c2 = 1 / c1  # Constant 2
    c3 = m * c2  # Constant 3
            
    # Mean and standard deviation
    if (pers == "Abel"):
        means = (1-(u/(c2-t)))*(b_t + (m*(1-t*c1))/(c1*(1-eval_mesh*c1)) - c3)
        sigmas = np.sqrt(u - (u**2)/(c2 - t))
    else:
        means = b_t + (u/(1-t))*(m-b_t)
        sigmas = np.sqrt((u/(1-t))**2*gamma**2+ (1-eval_mesh)*u/(1-t))    
    
    # Normal distribution functions
    norm_cdf_sup = norm.cdf((b_tu - means) / sigmas)
    norm_pdf_sup = norm.pdf((b_tu - means) / sigmas)
    
    # f_t(u) for u that t+u is in eval_mesh
    f_sup = (1/(c2-eval_mesh))*(c3-(means * (1-norm_cdf_sup) + sigmas * norm_pdf_sup))
    
    return f_sup


## Perspectiva Abel
def normal_boundary_ABEL(mesh, m, gamma, tol=1e-3, max_iter=1000):
    """
    Compute the optimal stopping boundary using a fixed-point Picard iteration.

    Parameters:
    - mesh (np.array): Temporal grid of [0,1].
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - tol (float): Tolerance for the fixed point algorithm.
    - max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    - boundary (np.array): Optimal stopping boundary for the times in mesh.
    """
    N = len(mesh)-1 # Number of temporal steps removing the last step
    c1 = (1 - gamma**2) # Constant 1
    boundary = np.full(N, m/c1)  # Initialize boundary with initial guess
    delta = np.diff(mesh[:-1]) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:-1]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u
            
            f_sup = f_t_sup(m, gamma, t, b_t, u, eval_mesh, b_tu, pers = "Abel")

            # Compute integral using Trapezoidal rule
            integral = np.trapz(f_sup, eval_mesh)

            # Update boundary estimate
            boundary_new[i] = h_normal(t, b_t) - integral

        # Compute error and check for convergence
        e = compute_error(boundary_new, boundary)
        if iter % 100 == 0:
            print(iter, e)
        if e < tol:
            break
        boundary = boundary_new

    return np.append(boundary, m/c1)


## Perspectiva Eduardo
def normal_boundary_EDUARDO(mesh, m, gamma, tol=1e-3, max_iter=1000):
    """
    Compute the optimal stopping boundary using a fixed-point Picard iteration.

    Parameters:
    - mesh (np.array): Temporal grid of [0,1].
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - tol (float): Tolerance for the fixed point algorithm.
    - max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    - boundary (np.array): Optimal stopping boundary for the times in mesh.
    """
    N = len(mesh)-1 # Number of temporal steps removing the last step
    c1 = (1 - gamma**2) # Constant 1
    boundary = np.full(N, m/c1)  # Initialize boundary with initial guess
    delta = np.diff(mesh[:-1]) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:-1]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u
            
            f_sup = f_t_sup(m, gamma, t, b_t, u, eval_mesh, b_tu, pers = "Eduardo")

            # Compute integral using Trapezoidal rule
            integral = np.trapz(f_sup, eval_mesh)

            # Update boundary estimate
            boundary_new[i] = h_normal(t, b_t) - integral

        # Compute error and check for convergence
        e = compute_error(boundary_new, boundary)
        if iter % 100 == 0:
            print(iter, e)
        if e < tol:
            break
        if np.isnan(e):
          print(f"Break at {iter} iteration because the error is {e}")
          break
        boundary = boundary_new

    return np.append(boundary, m/c1)


# Theoretical normal step by step
## Perspectiva Abel
def normal_boundary_ABEL_step_by_step(mesh, m, gamma, tol=1e-3, max_iter=1000):
    """
    Compute the optimal stopping boundary using a fixed-point Picard iteration.

    Parameters:
    - mesh (np.array): Temporal grid of [0,1].
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - tol (float): Tolerance for the fixed point algorithm.
    - max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    - boundary (np.array): Optimal stopping boundary for the times in mesh.
    """
    N = len(mesh) # Number of temporal steps
    c1 = (1 - gamma**2) # Constant 1
    boundary = np.full(N, m/c1)  # Initialize boundary with initial guess
    delta = np.diff(mesh) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u
            
            f_sup = f_t_sup(m, gamma, t, b_t, u, eval_mesh, b_tu, pers = "Abel")

            # Compute integral using Trapezoidal rule
            integral = np.trapz(f_sup, eval_mesh)

            # Update boundary estimate
            boundary_new[i] = h_normal(t, b_t) - integral

        # Compute error and check for convergence
        e = compute_error(boundary_new, boundary)
        if iter % 100 == 0:
            print(iter, e)
        if e < tol:
            break
        # Visualization
        plt.ion()  # Modo interactivo
        fig, ax = plt.subplots()
        ax.clear()  # Limpiar el gráfico anterior
        ax.plot(mesh, boundary, marker='o')
        ax.set_xlabel("mesh")
        ax.set_ylabel("boundary")
        ax.set_title(f"Iteration {iter}")
        plt.draw()
        plt.pause(1)  # Esperar 1 segundo
        
        plt.ioff()
        plt.show()
        
        # Update
        boundary = boundary_new

    return boundary


## Perspectiva Eduardo
def normal_boundary_EDUARDO_step_by_step(mesh, m, gamma, tol=1e-3, max_iter=1000):
    """
    Compute the optimal stopping boundary using a fixed-point Picard iteration.

    Parameters:
    - mesh (np.array): Temporal grid of [0,1].
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - tol (float): Tolerance for the fixed point algorithm.
    - max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    - boundary (np.array): Optimal stopping boundary for the times in mesh.
    """
    N = len(mesh) # Number of temporal steps
    c1 = (1 - gamma**2) # Constant 1
    c2 = 1 / c1  # Constant 2
    c3 = m * c2  # Constant 3
    boundary = np.full(N, c3)  # Initialize boundary with initial guess
    delta = np.diff(mesh) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u
            
            f_sup = f_t_sup(m, gamma, t, b_t, u, eval_mesh, b_tu, pers = "Eduardo")

            # Compute integral using Trapezoidal rule
            integral = np.trapz(f_sup, eval_mesh)

            # Update boundary estimate
            boundary_new[i] = h_normal(t, b_t) - integral

        # Compute error and check for convergence
        e = compute_error(boundary_new, boundary)
        if iter % 100 == 0:
            print(iter, e)
        if e < tol:
            break
        if np.isnan(e):
          print(f"Break at {iter} iteration because the error is {e}")
          break
        # Visualization
        plt.ion()  # Modo interactivo
        fig, ax = plt.subplots()
        ax.clear()  # Limpiar el gráfico anterior
        ax.plot(mesh, boundary, marker='o')
        ax.set_xlabel("mesh")
        ax.set_ylabel("boundary")
        ax.set_title(f"Iteration {iter}")
        plt.draw()
        plt.pause(1)  # Esperar 1 segundo
        
        plt.ioff()
        plt.show()
        
        # Update
        boundary = boundary_new

    return boundary
