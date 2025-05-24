# Libraries
import numpy as np

# Brownian bridges simulation
def simulate_brownian_bridge(t, z_t, T, z_T, u=None, n_steps=1000):
    """

    Simulates a Brownian bridge process between two fixed points.

    Parameters:
    - t (float): Starting time of the process. Must satisfy 0 <= t < T.
    - z_t (float): Position of the process at time t.
    - T (float): Ending time of the process.
    - z_T (float): Position of the process at time T.
    - u (float): If provided, the step size for the point of interest. Must satisfy t + u <= T.
    - n_steps (int): Number of steps for full path simulation (used only if u is None).

    Returns:
    - next_x (float): If `u` is provided, returns the next value of the Brownian bridge at time t + u.
    - all_times (np.array): If `u` is None, returns an array of time points between t and T.
    - path (np.array): If `u` is None, returns the full simulated Brownian bridge path.

    """
    
    assert 0 <= t < T, "t must be in [0, T)"
    if u is not None:
      assert t < t+u <= T, "t + u must be in (t, T]"
      mean = z_t + u*(z_T - z_t)/(T - t)
      var = u * (T - t - u) / (T - t)
      std = np.sqrt(var)
      next_x = np.random.normal(mean, std)

      return next_x

    else:
      all_times = np.linspace(t, T, n_steps)

      path = [z_t]
      current_t = t
      current_x = z_t

      for next_t in all_times[1:]:
        u = next_t - current_t
        mean = current_x + u*(z_T - current_x)/(T-current_t)
        var = u * (T - current_t -u) / (T - current_t)
        std = np.sqrt(var)
        next_x = np.random.normal(mean, std)
        path.append(next_x)

        current_t = next_t
        current_x = next_x

      return all_times, np.array(path)
  