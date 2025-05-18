# Libraries
import numpy as np
from scipy.stats import rv_discrete, rv_continuous, norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
import random


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
    N = len(mesh) # Number of temporal steps
    c1 = (1 - gamma**2) # Constant 1
    c2 = 1 / c1  # Constant 2
    c3 = m * c2  # Constant 3
    boundary = np.full(N, c3)  # Initialize boundary with initial guess
    delta = np.diff(mesh) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    def compute_error(boundary_new, boundary):
        """Computes relative error between iterations for convergence check."""
        numerator = np.sum((boundary_new - boundary)**2)
        denominator = np.sum(boundary_new**2)
        return np.sqrt(numerator / denominator) if denominator != 0 else float('inf')

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u

            means = (1-(u/(c2-t)))*(b_t + (m*(1-t*c1))/(c1*(1-eval_mesh*c1)) - c3)
            sigmas = np.clip(np.sqrt(u - (u**2)/(c2 - t)), 1e-8, None)


            norm_cdf_sup = norm.cdf((b_tu - means) / sigmas)
            norm_pdf_sup = norm.pdf((b_tu - means) / sigmas)

            f_sup = (1/(c2-eval_mesh))*(c3-(means * (1-norm_cdf_sup) + sigmas * norm_pdf_sup))

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

    return boundary


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
    N = len(mesh) # Number of temporal steps
    c1 = (1 - gamma**2) # Constant 1
    c2 = 1 / c1  # Constant 2
    c3 = m * c2  # Constant 3
    boundary = np.full(N, c3)  # Initialize boundary with initial guess
    eps = 1e-8  # Small value to prevent division errors
    delta = np.diff(mesh) # Array with the lenght of each step

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1)

    def compute_error(boundary_new, boundary):
        """Computes relative error between iterations for convergence check."""
        numerator = np.sum((boundary_new - boundary)**2)
        denominator = np.sum(boundary_new**2)
        return np.sqrt(numerator / denominator) if denominator != 0 else float('inf')

    for iter in range(max_iter):
        boundary_new = boundary.copy()

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh
            t = mesh[i]
            b_t = boundary[i] # b(t)
            eval_mesh = mesh[i+1:]  # Consider future times
            u = delta[i:]
            b_tu = boundary[i+1:] # b(t+u) for all u

            means = b_t + (u/(1-t))*(m-b_t)
            sigmas = np.sqrt((u/(1-t))**2*gamma**2+ (1-eval_mesh)*u/(1-t))

            norm_cdf_sup = norm.cdf((b_tu - means) / sigmas)
            norm_pdf_sup = norm.pdf((b_tu - means) / sigmas)

            f_sup = (1/(c2-eval_mesh))*(c3-(means * (1-norm_cdf_sup) + sigmas * norm_pdf_sup))

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

    return boundary


