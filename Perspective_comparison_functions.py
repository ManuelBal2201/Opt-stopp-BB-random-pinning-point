# Libraries
import numpy as np
from scipy.stats import rv_discrete, rv_continuous, norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
import random


# Brownian bridges simulation
def simulate_brownian_bridge(t, z_t, T, z_T, u=None, n_steps=1000):
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
  