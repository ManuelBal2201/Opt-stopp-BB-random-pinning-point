# Libraries
## Mathematics
import numpy as np
from scipy.interpolate import interp1d

## Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def OSB_visualisation(N, value_function, X_vals, title, name, exact_boundary = None):
    r"""
    
    Plot the continuation and stopping region of the OSB given the value function in each time and value of the respective grid.
    
    Parameters:
    - N (int): Number of temporal steps.
    - value_function (np.array): Value function for all the spatial points on each time step.
    - X_vals (np.array): Spatial grid we are considering.
    - title (string): Title of the plot.
    - name (string): Name of the file where the plot is saved.
    - exact_boundary (np.array): Exact boundary for plottin over the regions.
    
    Return:
    - plot: It is plotted the continuation region in green and the stopping region in red. It is saved in the file 'name'.
    
    """
    
    L = len(X_vals)
    t_mesh = np.linspace(0, 1, N)
    
    if N != L:
        # Adjust meshgrid for different dimensions
        T, X = np.meshgrid(t_mesh, np.linspace(np.min(X_vals), np.max(X_vals), L))
    
        # Interpole value_function to adjust `t_mesh`
        interp_func = interp1d(t_mesh, value_function, axis=0, kind='linear', fill_value="extrapolate")
        value_function_interp = interp_func(t_mesh)
    
        # Adjust X_vals to match value_function_interp
        X_grid = np.linspace(np.min(X_vals), np.max(X_vals), L)
        comparison = value_function_interp <= np.tile(X_grid, (N, 1))
        
        # Define colors
        cmap = mcolors.ListedColormap(['green', 'red'])
        
        # Visualization
        plt.figure(figsize=(8, 6))
        if exact_boundary is not None:
            plt.plot(t_mesh, exact_boundary, 'b-', label="Exact boundary")
        plt.pcolormesh(T, X, comparison.T, cmap=cmap, shading='auto')
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title(title)
        if exact_boundary is not None:
            plt.legend()
        plt.xlim(0, 1)
        plt.ylim(np.min(X_vals), np.max(X_vals))
        plt.savefig(name, dpi=300)
        plt.show()
        
    else:
        # Adjust meshgrid
        T, X = np.meshgrid(t_mesh, X_vals)
        
        # Define colors and mapping
        comparison = value_function <= np.tile(X_vals, (N, 1)) # Boolean matrix
        colors = np.where(comparison, 1, 0)  # 1 for True (red), 0 for False (green))
        cmap = plt.cm.colors.ListedColormap(['green', 'red'])
        
        # Visualization
        plt.figure(figsize=(8, 6))
        if exact_boundary is not None:
            plt.plot(t_mesh, exact_boundary, 'b-', label="Exact boundary")
        plt.pcolormesh(T, X, comparison, cmap=cmap, shading='auto')
        plt.xlabel("t")
        plt.ylabel("z")
        plt.title(title)
        if exact_boundary is not None:
            plt.legend()
        plt.xlim(0, 1)
        plt.ylim(np.min(X_vals), np.max(X_vals))
        plt.savefig(name, dpi=300)
        plt.show()