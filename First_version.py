# Libraries
## Mathematics
import numpy as np
from scipy.stats import rv_discrete, rv_continuous
from scipy.integrate import quad
from scipy.interpolate import interp1d

def h_definition(pdf, support, is_discrete = False, x_vals = None, probs = None):
  """

  Define the function given in (2.7).
  
  Parameters:
  - pdf (function):  Probability density function of X if it is continuous.
  - support (list): Support of X if it is continuous. The first element is the lower bound an d the second is the upper bound.
  - is_discrete (bool): Indicates if X is discrete.
  - x_vals (array): Support of X if it is discrete.
  - probs (array): Probabilities of x_vals if X is discrete.

  Return:
  - h (función): Function h for the particular pdf or pmf.

  """

  def h(t, z):
      if is_discrete:
        exponential = np.exp((x_vals * z) / (1 - t) - (t * x_vals**2) / (2 * (1 - t)))
        num_integral = np.sum(x_vals * exponential * probs)
        den_integral = np.sum(exponential * probs)

      else:
        num_integral, _ = quad(lambda u: u * np.exp((u * z) / (1 - t) - (t * u**2) / (2 * (1 - t))) * pdf(u), support[0], support[1])
        den_integral, _ = quad(lambda u: np.exp((u * z) / (1 - t) - (t * u**2) / (2 * (1 - t))) * pdf(u), support[0], support[1])

      return num_integral / den_integral
  return np.vectorize(h)


def spatial_grid(mu, L = 100):
  """

  Spatial grid that it is used as initial values to simulate the process.

  Parameters:
  - mu: It is the distribution of X.
    - If X is discrete, is rv_discrete.
    - If X is continuous, is rv_continuous.
  - L (int): Number of points on the spatial grid. It is needed that L>5.

  Return:
  - X_vals: X values we are considering.

  """

  if isinstance(mu, rv_discrete):  # If X is discrete.
    x_vals = mu.xk  # Support of X.
    X_vals = np.linspace(np.min(x_vals), np.max(x_vals) + 0.839924, L)  # Spatial grid.

  else:  # If X is continuos.
    (a, b) = mu.support() # Extract the support interval.
    if np.isinf(a) and not np.isinf(b): # If it is not bounded below.
      X_vals_unsorted = np.append(mu.rvs(size = L-5), np.linspace(b, b + 0.839924, 5)) # Generate randomly L-5 values following a mu pdf and a grid of 5 points between b and b+beta.
      X_vals = np.sort(X_vals_unsorted) # Sort the values.

    elif np.isinf(b): # If it is unbounded or not bounded above.
      X_vals = np.sort(mu.rvs(size = L)) # Sort the values.

    else: # If it is bounded.
      X_vals = np.linspace(a, b + 0.839924, L)  # Spatial grid.
      # CAMBIARLO POR GENERAR ACORDE A LA DENSIDAD?
      #X_vals_unsorted = np.append(mu.rvs(size = L-5), np.linspace(b, b + 0.839924, 5)) # Generate randomly L-5 values following a mu pdf and a grid of 5 points between b and b+beta.
      #X_vals = np.sort(X_vals_unsorted) # Sort the values.

  return X_vals


def simulate_process(z, M, t, dt, h):
  """

  Simulation of Z_{t + \Delta t} using Euler-Maruyama estimation.

  Parameters:
  - z (float): Value of Z_t.
  - M (int): Number of Monte Carlo simulations.
  - t (float): Value of t (temporal variable). t is in (0, 1).
  - dt (float): Temporal step length.
  - h (function): h function that appears in the Z_{t + \Delta t} definition.

  Return:
  - paths: M possible paths of the process.

  """

  # Simulations using Euler-Maruyama
  dW = np.random.randn(M)*np.sqrt(dt)  # Brownian motion of Z_{t + \Delta t}.
  drift = (h(t, z) - z)/(1 - t) # Drift of Z_{t + \Delta t}.
  paths = z + np.multiply(drift, dt) + dW # Array where the i-th element is value of Z_{t + \Delta t} for the i-th Monte Carlo iteration.

  return paths

def v_expectance(X_vals, M, t, dt, h_function, v):
  """

  E[V(t+s, Z_{t+s}) | Z_t = x_val] for each value x_val in X_vals.

  Parameters:
  - X_vals: Spatial grid we are considering.
  - M (int): Number of Monte Carlo simulations.
  - t (float): Value of t (temporal variable). t is in (0, 1).
  - dt (float): Temporal step length.
  - h_function (function): h function that appears in the Z_{t + \Delta t} definition.
  - v (np.array): Value function in t+dt. The i-th element is the value function in the i-th point of X_vals.

  Return:
  - v_expec (np.array): E[V(t+s, Z_{t+s}) | Z_t = x_val] for each x_val in X_vals. The i-th element is the expected value function in the i-th point of X_vals.

  """
  # Create an interpolator function.
  interp_func = interp1d(X_vals, v, kind='linear', fill_value="extrapolate")  # You can use 'linear', 'quadratic', 'cubic', etc.

  # Initialise E[V(t+s, Z_{t+s}) | Z_t = X_vals].
  v_expec = np.zeros(len(X_vals)) # Initialise the numpy array E[V(t+s, Z_{t+s}) | Z_t = X_vals].

  for i, x_val in enumerate(X_vals): # Loop for each value x_val of X_vals.
    Z_next = simulate_process(x_val, M, t, dt, h_function) # Simulations of the next step.
    v_new = interp_func(Z_next) # Value function for each "arrival".
    v_expec[i] = np.mean(v_new) # Expectance of value function for each "begining".

  return v_expec


def optimal_stopping_montecarlo(mu = rv_discrete(name = 'Delta Dirac', values = (0, 1)), N = 100, M = 50000, L = 100, m = 0, gamma= 1/2):
  """

  Implementation for the optimal stopping boundary with Monte Carlo.

  Parameters:
  - mu: It is the distribution of X.
    - If X is discrete, is rv_discrete.
    - If X is continuous, is rv_continuous.
  - N (int): Number of temporal steps.
  - M (int): Number of Monte Carlo simulations.
  - L (int): Number of spatial grid for Z_t.

  Return:
  - value_function (np.array): Value function for all the spatial points on each time step.
  - X_vals (np.array): Spatial grid we are considering.

  """

  # General initializations
  dt = 1/N  # Temporal step length.

  # Generation of the initial mesh for t = 1.
  if isinstance(mu, rv_discrete):  # If X is discrete.
    X_vals = spatial_grid(mu = mu, L = L) # Values of X that it is considered as initial values.
    h_function = h_definition(pdf = None, support = None, is_discrete = True, x_vals = mu.xk, probs = mu.pk) # Define the h(t, x) function.

  elif isinstance(mu, rv_continuous):  # If X is continuous.
    X_vals = spatial_grid(mu = mu, L = L) # Values of X that it is considered as initial values.
    h_function = h_definition(pdf = mu.pdf, support = mu.support()) # Define the h(t, x) function.

  else:
    raise ValueError("mu must be a rv_discrete or rv_continuous variable.")
  h_function = lambda t, z: (z*gamma**2 + m*(1-t))/(1-t*(1-gamma**2)) # BORRAR

  value_function = np.zeros((N, L))  # Initialise the array where the boundary points are saved.

  value_function[N-2, :] = X_vals # Value function in t = 1-dt.

  # Boundary obtention
  for j in range(2, N): # Loop for Z_{t + \Delta t}.
    t = 1 - (j*dt) # Time of the step.

    Expectance_V_next = v_expectance(X_vals, M, t, dt, h_function, value_function[N-j+1, :]) # Compute the value function expectance.
    value_function[N-j, :] = np.maximum(X_vals, Expectance_V_next) # Dynamic Principle

  return value_function, X_vals

