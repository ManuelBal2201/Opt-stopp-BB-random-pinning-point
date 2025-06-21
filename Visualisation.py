# Libraries
## Mathematics
import numpy as np
from scipy.interpolate import interp1d

## Visualisation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Standard case (theoretical development and testing)
def OSB_visualisation(N, value_function, X_vals, title, name, exact_boundary = None, a = None, b = None):
    r"""
    
    Plot the continuation and stopping region of the OSB given the value function in each time and value of the respective grid.
    
    Parameters:
    - N (int): Number of temporal steps.
    - value_function (np.array): Value function for all the spatial points on each time step.
    - X_vals (np.array): Spatial grid we are considering.
    - title (string): Title of the plot.
    - name (string): Name of the file where the plot is saved.
    - exact_boundary (np.array): Exact boundary for plottin over the regions.
    - a (float): Y axis lower bound for plotting. If None, then a = min(X_vals).
    - b (float): Y axis upper bound for plottin. If None, then b = max(X_vals).
    
    Return:
    - plot: It is plotted the continuation region in green and the stopping region in red. It is saved in the file 'name'.
    
    """
    
    # Initialise variables
    L = len(X_vals)
    t_mesh = np.linspace(0, 1, N)
    if a is None: a = min(X_vals)
    if b is None: b = max(X_vals)    
    
    if N != L:
        # Adjust meshgrid for different dimensions
        T, X = np.meshgrid(t_mesh, X_vals)
    
        # Interpole value_function to adjust `t_mesh`
        interp_func = interp1d(t_mesh, value_function, axis=0, kind='linear', fill_value="extrapolate")
        value_function_interp = interp_func(t_mesh)
    
        # Adjust X_vals to match value_function_interp
        comparison = value_function_interp <= np.tile(X_vals, (N, 1))
        
        # Define colors
        cmap = mcolors.ListedColormap(['green', 'red'])
        
        # Visualisation
        plt.figure(figsize=(8, 6)) # Plot size
        if exact_boundary is not None: # If the exact boundary is known and it is wanted to plot over the regions.
            plt.plot(t_mesh, exact_boundary, 'b-', label="Exact boundary")
        plt.pcolormesh(T, X, comparison.T, cmap=cmap, shading='auto') # Stopping and continuation regions
        plt.xlabel("t") # X axis label
        plt.ylabel("z") # Y axis label
        plt.title(title) # Assign title
        if exact_boundary is not None: # Legend if exact boundary is plotted
            plt.legend() 
        plt.xlim(0, 1) # X axis limits
        plt.ylim(a, b) # Y axis limits
        plt.savefig(name, dpi=300) # Save plot
        plt.show() # Show plot
        
    else:
        # Adjust meshgrid
        T, X = np.meshgrid(t_mesh, X_vals)
        
        # Define colors and mapping
        comparison = value_function <= np.tile(X_vals, (N, 1)) # Boolean matrix
        cmap = plt.cm.colors.ListedColormap(['green', 'red'])
        
        # Visualisation
        plt.figure(figsize=(8, 6)) # Plot size
        if exact_boundary is not None: # If the exact boundary is known and it is wanted to plot over the regions.
            plt.plot(t_mesh, exact_boundary, 'b-', label="Exact boundary") 
        plt.pcolormesh(T, X, comparison, cmap=cmap, shading='auto') # Stopping and continuation regions
        plt.xlabel("t") # X axis label
        plt.ylabel("z") # Y axis label
        plt.title(title) # Assign title
        if exact_boundary is not None: # Legend if exact boundary is plotted
            plt.legend() 
        plt.xlim(0, 1) # X axis limits
        plt.ylim(a, b) # Y axis limits
        plt.savefig(name, dpi=300) # Save plot
        plt.show() # Show plot
        
        
# Practical case     
def OSB_visualisation_practical_case(N, value_function, X_vals, volatility, Z_0, title, name, t_0 = None, a = None, b = None):
    r"""
    
    Plot the continuation and stopping region of the OSB given the value function in each time and value of the respective grid.
    
    Parameters:
    - N (int): Number of temporal steps.
    - value_function (np.array): Value function for all the spatial points on each time step.
    - X_vals (np.array): Spatial grid we are considering.
    - volatility (float): Volatility of the process.
    - Z_0 (float): First observation of the process.
    - title (string): Title of the plot.
    - name (string): Name of the file where the plot is saved.
    - t_0 (float): X axis lower bound for plotting. If None, then t_0 = 0.
    - a (float): Y axis lower bound for plotting. If None, then a = min(X_vals).
    - b (float): Y axis upper bound for plottin. If None, then b = max(X_vals).
    
    Return:
    - plot: It is plotted the continuation region in green and the stopping region in red. It is saved in the file 'name'.
    
    """
    
    # Initialise variables
    L = len(X_vals)
    t_mesh = np.linspace(0, 1, N) 
    if t_0 is None: t_0 = 0 
    
    if N != L:
        # Adjust meshgrid for different dimensions
        T, X = np.meshgrid(t_mesh, X_vals)
    
        # Interpole value_function to adjust `t_mesh`
        interp_func = interp1d(t_mesh, value_function, axis=0, kind='linear', fill_value="extrapolate")
        value_function_interp = interp_func(t_mesh)
    
        # Adjust X_vals to match value_function_interp
        comparison = value_function_interp <= np.tile(X_vals, (N, 1))
        
        # Define colors
        cmap = mcolors.ListedColormap(['green', 'red'])
        
        # Scale data
        X = X*volatility + Z_0
        X_vals = X_vals*volatility + Z_0
        if a is None: a = min(X_vals)
        if b is None: b = max(X_vals) 
        
        # Visualisation
        plt.figure(figsize=(8, 6)) # Plot size
        plt.pcolormesh(T, X, comparison.T, cmap=cmap, shading='auto') # Stopping and continuation regions
        plt.xlabel("t") # X axis label
        plt.ylabel("z") # Y axis label
        plt.title(title) # Assign title
        plt.xlim(t_0, 1) # X axis limit
        plt.ylim(a, b) # Y axis limit
        plt.savefig(name, dpi=300) # Save plot
        plt.show() # Show plot
        
    else:
        # Adjust meshgrid
        T, X = np.meshgrid(t_mesh, X_vals)
        
        # Define colors and mapping
        comparison = value_function <= np.tile(X_vals, (N, 1)) # Boolean matrix
        cmap = plt.cm.colors.ListedColormap(['green', 'red'])
        
        # Scale data
        X = X*volatility + Z_0
        X_vals = X_vals*volatility + Z_0
        if a is None: a = min(X_vals)
        if b is None: b = max(X_vals) 
        
        # Visualisation
        plt.figure(figsize=(8, 6)) # Plot size
        plt.pcolormesh(T, X, comparison, cmap=cmap, shading='auto') # Stopping and continuation regions
        plt.xlabel("t") # X axis label
        plt.ylabel("z") # Y axis label
        plt.title(title) # Assign title
        plt.xlim(t_0, 1) # X axis limit
        plt.ylim(a, b) # Y axis limit
        plt.savefig(name, dpi=300) # Save plot
        plt.show() # Show plot