# Libraries
import numpy as np
from scipy.stats import rv_discrete, rv_continuous
from scipy.integrate import quad
from scipy.interpolate import interp1d


# h definition
def h_definition(pdf, support, is_discrete = False, x_vals = None, probs = None):
  """

  Define the function given in Equation 2.7. REVISAR SI EL INDICE ES CORRECTO EN EL DEFINITIVO

  Parameters:
  - pdf (function):  Probability density function of X if it is continuous.
  - support (list): Support of X if it is continuous. The first element is the lower bound an d the second is the upper bound.
  - is_discrete (bool): Indicates if X is discrete.
  - x_vals (np.array): Support of X if it is discrete.
  - probs (np.array): Probabilities of x_vals if X is discrete.

  Return:
  - h (function): Function h for the particular pdf or pmf.

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


# Spatial grid definition
def spatial_grid(mu, h, L = 100):
  """

  Spatial grid that it is used as initial values to simulate the process.

  Parameters:
  - mu: It is the distribution of X.
    - If X is discrete, is rv_discrete.
    - If X is continuous, is rv_continuous.
  - h (function): Function h for the particular pdf of mu.
  - L (int): Number of points on the spatial grid. It is needed that L>5.

  Return:
  - X_vals (np.array): X values we are considering.

  """

  if isinstance(mu, rv_discrete):  # If X is discrete.
    x_vals = mu.xk  # Support of X.
    X_vals = np.linspace(np.min(x_vals), np.max(x_vals) + 0.839924, L)  # Spatial grid.

  else:  # If X is continuos.
    (a, b) = mu.support() # Extract the support interval.
    if np.isinf(a) and not np.isinf(b): # If it is not bounded below.
      X_vals_random = mu.rvs(size = L*0.97) # Generate randomly L*0.97 values following a mu pdf.
      if (max(X_vals_random)-min(X_vals_random) < 0.839924): # The distance between the maximum and minimum is very low.
          mean = mu.mean()
          a = mean
          b = mean
          std = mu.std()
          while True: # Loop until obtaining an interval where h changes sign.
              a = a - std - 1 # Substract 1 to improve pacing for low standard deviations.
              x_vals = np.linspace(a, b + 0.839924, 1000)
              h_vals = [h(1,x) for x in x_vals]
              sign_changes = np.where(np.diff(np.sign(h_vals)))[0]
              if sign_changes.size > 0:
                  break
          X_vals = np.linspace(a, b + 0.839924, L)
      else:
          X_vals_eq = np.linspace(b, b + 0.839924, L*0.03) # Generate L*0.03 equidistant points between b and b+beta.
          X_vals = np.sort(np.append(X_vals_random, X_vals_eq)) # Sort the values.

    elif np.isinf(b): # If it is unbounded or not bounded above.
      X_vals_random = mu.rvs(size = L) # Generate randomly L*0.97 values following a mu pdf.
      if (max(X_vals_random)-min(X_vals_random) < 0.839924): # The distance between the maximum and minimum is very low.
          mean = mu.mean()
          a = mean
          b = mean
          std = mu.std()
          while True: # Loop until obtaining an interval where h changes sign.
              a = a - std - 1 # Substract 1 to improve pacing for low standard deviations.
              b = b + std + 1 # Add 1 to improve pacing for low standard deviations.
              x_vals = np.linspace(a, b, 1000)
              h_vals = [h(1,x) for x in x_vals]
              sign_changes = np.where(np.diff(np.sign(h_vals)))[0]
              if sign_changes.size > 0:
                  break
          X_vals = np.linspace(a, b, L)
      else:
          X_vals = np.sort(X_vals_random) # Sort the values.

    else: # If it is bounded.
      X_vals = np.linspace(a, b + 0.839924, L)  # Spatial grid.

  return X_vals


# Process simulation
def simulate_process(z, M, t, dt, h):
  r"""

  Simulation of Z_{t + \Delta t} using Euler-Maruyama estimation.

  Parameters:
  - z (float): Value of Z_t.
  - M (int): Number of Monte Carlo simulations.
  - t (float): Value of t (temporal variable). t is in (0, 1).
  - dt (float): Temporal step length.
  - h (function): h function that appears in the Z_{t + \Delta t} definition.

  Return:
  - paths (np.array): M possible paths of the process.

  """

  # Simulations using Euler-Maruyama
  dW = np.random.randn(M)*np.sqrt(dt)  # Brownian motion of Z_{t + \Delta t}.
  drift = (h(t, z) - z)/(1 - t) # Drift of Z_{t + \Delta t}.
  paths = z + np.multiply(drift, dt) + dW # Array where the i-th element is value of Z_{t + \Delta t} for the i-th Monte Carlo iteration.

  return paths


# First step value function (backwards loop)
def value_function_first_step(mu, X_vals, M, dt, h_function):
  r"""

  Value function in t = 1-dt.

  Parameters:
  - mu: It is the distribution of X.
    - If X is discrete, is rv_discrete.
    - If X is continuous, is rv_continuous.
  - X_vals: Spatial grid we are considering.
  - M (int): Number of Monte Carlo simulations.
  - dt (float): Temporal step length.
  - h_function (function): h function that appears in the Z_{t + \Delta t} definition.

  Return:
  - v (np.array): Value function in t = 1-dt. The i-th element is the value function in the i-th point of X_vals.

  """

  if isinstance(mu, rv_discrete):  # If X is discrete.
      x_vals = mu.xk # mu support.
      
      if len(x_vals) == 1: # Dirac's delta case
          Z_next = np.array([simulate_process(x_val, M, 1-dt, dt, h_function) for x_val in X_vals]) # Obtain Z values for the next step.
          Expectance_V_next = np.mean(Z_next, axis=1) # Obtain the expenctance of v in the next step.
          v = np.maximum(X_vals, Expectance_V_next) # Dynamic principle.
          
          v[:np.argmax(v == X_vals)] = x_vals
      
      else:
          interp_func = interp1d(x_vals, x_vals, kind='linear', fill_value="extrapolate")
          v = interp_func(X_vals)
          
  
  else:  # If X is continuos.
      v = X_vals

  return v


# Value function expectance
def v_expectance_1(X_vals, M, t, dt, h_function, v):
  r"""

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
  interp_func = interp1d(X_vals, v, kind='linear', fill_value="extrapolate")

  # Initialize E[V(t+s, Z_{t+s}) | Z_t = X_vals].
  v_expec = np.zeros(len(X_vals)) # Initialize the numpy array E[V(t+s, Z_{t+s}) | Z_t = X_vals].

  for i, x_val in enumerate(X_vals): # Loop for each value x_val of X_vals.
    Z_next = simulate_process(x_val, M, t, dt, h_function) # Simulations of the next step.
    v_new = interp_func(Z_next) # Value function for each "arrival".
    v_expec[i] = np.mean(v_new) # Expectance of value function for each "begining".

  return v_expec


# Optimal stopping boundary simulation
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
    h_function = h_definition(pdf = None, support = None, is_discrete = True, x_vals = mu.xk, probs = mu.pk) # Define the h(t, x) function.
    X_vals = spatial_grid(mu = mu, h = h_function, L = L) # Values of X that it is considered as initial values.
    
  elif isinstance(mu, rv_continuous):  # If X is continuous.
    #h_function = h_definition(pdf = mu.pdf, support = mu.support()) # Define the h(t, x) function.
    h_function = lambda t, z: (z*gamma**2 + m*(1-t))/(1-t*(1-gamma**2)) # BORRAR
    X_vals = spatial_grid(mu = mu, h = h_function, L = L) # Values of X that it is considered as initial values.

  else:
    raise ValueError("mu must be a rv_discrete or rv_continuous variable.")
  

  value_function = np.zeros((N, L))  # Initialize the array where the boundary points are saved.

  value_function[N-1, :] = value_function_first_step(mu, X_vals, M, dt, h_function) # Value function in t = 1-dt.

  # Boundary obtention
  for j in range(2, N): # Loop for Z_{t + \Delta t}.
    t = 1 - (j*dt) # Time of the step.

    Expectance_V_next = v_expectance(X_vals, M, t, dt, h_function, value_function[N-j+1, :]) # Compute the value function expectance.
    value_function[N-j, :] = np.maximum(X_vals, Expectance_V_next) # Dynamic Principle

  return value_function, X_vals