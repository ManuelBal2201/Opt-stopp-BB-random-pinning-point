# Libraries
import numpy as np
from scipy.stats import norm


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
def f_t_sup(m, gamma, t, b_t, u, b_tu):
    """
    f_t evaluated in the upper limits of the integrals.
    
    Parameters:
    - m (float): Mean of the distribution.
    - gamma (float): Standard deviation of the distribution.
    - t (float): Actual time t_n.
    - b_t (float): Actual value of the boundary.
    - u (float): Step that it is considered.
    - b_tu (float): Boundary value for eval_mesh times.
    
    Returns:
    - f_sup (np.array): f_t evaluated in the upper limits of the integrals.
    
    """
    c1 = (1 - gamma**2) # Constant 1
    c2 = 1 / c1  # Constant 2
    c3 = m * c2  # Constant 3
            
    # Mean and standard deviation
    mean = b_t + u/(1 - t*c1)*(m - b_t*c1)
    sigma = np.sqrt(u - c1*u**2/(1 - t*c1))
    
    # Normal distribution functions
    norm_cdf_sup = norm.cdf((b_tu - mean) / sigma)
    norm_pdf_sup = norm.pdf((b_tu - mean) / sigma)
    
    # f_t(u)
    f_sup = (1/(c2-t-u))*(c3-(mean * (1-norm_cdf_sup) + sigma * norm_pdf_sup))
    
    return f_sup


## Optimal stopping boundary
def optimal_stopping_Normal(mesh, m, gamma, tol=1e-3, max_iter=1000):
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
    N = len(mesh)-1 # Number of temporal steps removing the last step.
    c1 = (1 - gamma**2) # Constant 1.
    boundary = np.full(N, m/c1)  # Initialize boundary with initial guess.

    h_normal = lambda t, z: (z * gamma**2 + m * (1 - t)) / (1 - t * c1) # h definition for this particular case.

    for iter in range(max_iter):
        boundary_new = boundary.copy() # Initialize the boundary as first one.

        for i in range(N - 2, -1, -1):  # Iterate backwards over the mesh.
            t = mesh[i] # Actual time.
            b_t = boundary[i] # b(t).
            eval_mesh = mesh[i+1:-1]  # Consider future times.
            u = eval_mesh-t # Step size from t to eval_mesh.
            b_tu = boundary[i+1:] # b(t+u) for all u.
            
            f_sup = f_t_sup(m, gamma, t, b_t, u, b_tu) # f_t(u) for all u.

            # Compute integral using Trapezoidal rule.
            integral = np.trapz(f_sup, eval_mesh)

            # Update boundary estimate.
            boundary_new[i] = h_normal(t, b_t) - integral

        # Compute error and check for convergence.
        e = compute_error(boundary_new, boundary)
        if iter % 100 == 0:
            print(iter, e)
        if e < tol:
            break
        boundary = boundary_new # Update boundary for the next iteration.

    return np.append(boundary, m/c1) # Add b(1).