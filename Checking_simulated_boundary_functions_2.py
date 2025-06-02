# Libraries
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.interpolate import interp1d

# Process simulation
def simulate_brownian_bridge(t, z_t, T, z_T, u):
    r"""

    Simulates a Brownian bridge process between two fixed points.

    Parameters:
    - t (float): Starting time of the process. Must satisfy 0 <= t < T.
    - z_t (float): Position of the process at time t.
    - T (float): Ending time of the process.
    - z_T (np.array or float): Position of the process at time T.
    - u (float): If provided, the step size for the point of interest. Must satisfy t + u <= T.

    Returns:
    - next_x (float): If `u` is provided, returns the next value of the Brownian bridge at time t + u.

    """
    
    # Brownian bridge step.
    mean = z_t + u*(z_T - z_t)/(T - t)
    var = u * (T - (t + u)) / (T - t)
    std = np.sqrt(var)
    next_x = np.random.normal(loc = mean, scale = std)
  
    return next_x

def mu_ty_simulator(mu, weights, parameters, y, M, t):
    r"""
    
    Generate samples of \mu_{t,y}.
    
    Parameters:
    - mu (string): Type of mixture considered
        - If X is discrete, is "discrete".
        - If X is continuous, is "continuous".
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array.
        - If X is discrete, the first dimension are the points where the probability is positive, the second one are the probabilities.
        - If X is continuous, the first dimension is the mean, the second one is the variance.
    - y: Point of spatial grid that considered.
    - M (int): Number of samples.
    - t (float): Value of t (temporal variable).
    
    Return:
    - mu_ty (np.array): Samples of \mu_{t,y}.
    
    """
    if mu == "discrete":
        # Initialize variables
        points = parameters[0]
        probabilities = parameters[1]
        
        # Update probabilities
        probabilities_ty_numerator = np.exp(points*y/(1-t)-t*points**2/(2*(1-t)))*probabilities
        probabilities_ty = probabilities_ty_numerator/np.sum(probabilities_ty_numerator)
        
        # Sample distributions
        selection = np.random.choice(list(range(0,parameters.shape[1])), size = M, p = probabilities_ty)
        
        # Samples of \mu_{t,z}
        mu_ty = points[selection]
    elif mu == "continuous":
        # Initialize variables
        m = parameters[0]
        gamma2 = parameters[1]
        
        # Update parameters and weights to t,z
        A = t/(2*(1 - t)) + 1/(2*gamma2)
        B = (y/(1-t)) + m/gamma2
        C = B**2/(4*A) - m**2/(2*gamma2)
        
        m_tz = B/(2*A)
        gamma_tz = np.sqrt(1/(2*A))
        if np.isinf(np.exp(C)).any(): # If there is an element np.exp(C), it is the only term in the expression of weight_ty of the same order as the denominator in the numerator 
            weights_ty = (np.exp(C) == np.inf).astype(int)
        else:
            weights_ty_numerator = weights * np.exp(C) * np.sqrt(np.pi/A)
            weights_ty = weights_ty_numerator/np.sum(weights_ty_numerator)
        
        # Sample distributions
        selection = np.random.choice(list(range(0,parameters.shape[1])), size = M, p = weights_ty)
        
        # Samples of \mu_{t,z}
        mu_ty = np.random.normal(loc=m_tz[selection], scale=gamma_tz[selection])
        
    else:
        raise ValueError("mu must be a 'discrete' or 'continuos'.")
        
    return mu_ty


# Value function expectance
## Define function to run in parallel
def compute_v_expec(mu, weights, parameters, x_val, M, t, u, interp_func):
    r"""
    
    Compute E[V(t+s, Z_{t+s}) | Z_t = x_val].
    
    Parameters:
    - mu (string): Type of mixture considered
        - If X is discrete, is "discrete".
        - If X is continuous, is "continuous".
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array.
        - If X is discrete, the first dimension are the points where the probability is positive, the second one are the probabilities.
        - If X is continuous, the first dimension is the mean, the second one is the variance.
    - x_val: Point of spatial grid that considered.
    - M (int): Number of Monte Carlo simulations.
    - t (float): Value of t (temporal variable).
    - u (float): Temporal step length.
    - v (np.array): Value function in t+dt. The i-th element is the value function in the i-th point of X_vals.
    
    Return:
    - v_expec (np.array): E[V(t+s, Z_{t+s}) | Z_t = x_val].
    
    """
    # Generate random points where the process may end
    mu_ty_points = mu_ty_simulator(mu = mu, weights = weights, parameters = parameters, y = x_val, M = M, t = t)
    
    # Simulate a brownian bridge that ends in the mu_tz_points. Keep only the value on t+u
    Z_tu = simulate_brownian_bridge(t, z_t = x_val, T = 1, z_T = mu_ty_points, u = u)
    
    # Obtaining the value function for the obtained points
    v_new = interp_func(Z_tu)
    
    # Compute E[V(t+s, Z_{t+s}) | Z_t = x_val]
    v_expec = np.mean(v_new)
    
    return v_expec 

## Wrapper function to parallelize
def parallel_loop(mu, weights, parameters, X_vals, M, t, u, interp_func):
    r"""
    
    Wrapper function to parallelize.
    
    Parameters:
    - mu (string): Type of mixture considered
        - If X is discrete, is "discrete".
        - If X is continuous, is "continuous".
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array.
        - If X is discrete, the first dimension are the points where the probability is positive, the second one are the probabilities.
        - If X is continuous, the first dimension is the mean, the second one is the variance.
    - x_val: Point of spatial grid that considered.
    - M (int): Number of Monte Carlo simulations.
    - t (float): Value of t (temporal variable).
    - u (float): Temporal step length.
    - v (np.array): Value function in t+dt. The i-th element is the value function in the i-th point of X_vals.
    
    Return:
    - v_expec (np.array): E[V(t+s, Z_{t+s}) | Z_t = x_val].
    
    """
    v_expec = np.zeros(len(X_vals))
    with ProcessPoolExecutor() as executor:
        # Use partial to freeze all parameters except x_val
        func = partial(compute_v_expec, mu, weights, parameters, M=M, t=t, u=u, interp_func=interp_func)
        results = executor.map(func, X_vals)
    v_expec[:] = list(results)
    return v_expec

## Compute E[V(t+s, Z_{t+s}) | Z_t = x_val] for each value x_val in X_vals
def v_expectance(mu, weights, parameters, X_vals, M, t, u, v):
    r"""
    
    Compute E[V(t+s, Z_{t+s}) | Z_t = x_val] for each value x_val in X_vals.
    
    Parameters:
    - mu (string): Type of mixture considered
        - If X is discrete, is "discrete".
        - If X is continuous, is "continuous".
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array.
        - If X is discrete, the first dimension are the points where the probability is positive, the second one are the probabilities.
        - If X is continuous, the first dimension is the mean, the second one is the variance.
    - X_vals: Spatial grid.
    - M (int): Number of Monte Carlo simulations.
    - t (float): Value of t (temporal variable).
    - u (float): Temporal step length.
    - v (np.array): Value function in t+dt. The i-th element is the value function in the i-th point of X_vals.
    
    Return:
    - v_expec (np.array): E[V(t+s, Z_{t+s}) | Z_t = x_val] for each x_val in X_vals. The i-th element is the expected value function in the i-th point of X_vals.
    
    """
    
    # Create an interpolator function
    interp_func = interp1d(X_vals, v, kind='linear', fill_value="extrapolate")
      
    # Initialize E[V(t+s, Z_{t+s}) | Z_t = X_vals]
    v_expec = np.zeros(len(X_vals))
    
    # Expected value function
    v_expec = parallel_loop(mu, weights, parameters, X_vals, M, t, u, interp_func)
    
    return v_expec

# Optimal stopping boundary simulation
def optimal_stopping_montecarlo(mu = "continuous", weights = np.array([1]), parameters = np.array([[0],[1]]), N = 100, a = -1, b = 1, L = 100, M = 5000, alpha = 1):
    """
    
    Implementation for the optimal stopping boundary with Monte Carlo.
    
    Parameters:
    - mu (string): Type of mixture considered
        - If X is discrete, is "discrete".
        - If X is continuous, is "continuous".
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array.
        - If X is discrete, the first dimension are the points where the probability is positive, the second one are the probabilities.
        - If X is continuous, the first dimension is the mean, the second one is the variance.
    - N (int): Number of temporal steps.
    - a (float): Lower bound of the spatial grid.
    - b (float): Upper bound of the spatial grid.
    - L (int): Number of spatial grid for Z_t.
    - M (int): Number of Monte Carlo simulations for simulating Z_{t+u} for each time and each point in the spatial grid.
    - alpha (float): Adaptivity factor for time-dependent Monte Carlo simulations. 
    
    Return:
    - value_function (np.array): Value function for all the spatial points on each time step.
    - X_vals (np.array): Spatial grid we are considering.
    
    """
    
    # General initializations
    t_mesh = np.linspace(0, 1, N) # Temporal grid
    u = np.diff(t_mesh)[0] # Temporal step length
    X_vals = np.linspace(a, b, L) # Spatial grid
    value_function = np.zeros((N, L))  # Initialize the array where the boundary points are saved
    value_function[N-2, :] = X_vals # Value function in t = 1
    
    # Obtaining the boundary
    for j in range(N-2, 0, -1): # Loop for Z_{t + u}.
        if j % 100 == 0:
            print(f"Temporal grid point: {j}")
        t = t_mesh[j-1] # Time of the step
        M_t = int(M * (1 + alpha * (1 - t)))
        
        Expectance_V_next = v_expectance(mu, weights, parameters, X_vals, M_t, t, u, value_function[j, :]) # Compute the value function expectance
        value_function[j-1, :] = np.maximum(X_vals, Expectance_V_next) # Dynamic Principle
    
    return value_function, X_vals