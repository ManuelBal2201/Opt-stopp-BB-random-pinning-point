# Libraries
import numpy as np

# EMV estimation

# KDE
def KDE(Z_1, bw_method = "Silverman"):
    r"""
    
    Kernel density estimation for the pinning point distribution.
    
    Parameters:
    - Z_1 (np.array): Array of historical terminal points.
    - bw_method (string or float): How bandwith is calculated. If string, only admitted "Silverman" and "Scott". If float, it is assigned that value.
    
    Return:
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array. 
    The first dimension is the mean, the second one is the standard deviation.
    
    """
    # Initalize variables
    n = len(Z_1)
    sigma = np.std(Z_1, ddof=1)
    
    # Compute the bandwidth
    if isinstance(bw_method, str):
        if bw_method == "Silverman":
            h = (4/3*(sigma**5/n))**(1/5)
        elif bw_method == "Scott":
            h = sigma/(n**(1/5))
        else:
            raise ValueError("If bw_method is a string, it must be 'Silverman' or 'Scott'.")
        
    elif isinstance(bw_method, float):
        h = bw_method
    else:
        raise ValueError("bw_method must be a string or float.")
    
    # Compute the weights and parameters of the gaussian mixture. They are thought to be directly introduced in optimal_stopping_montecarlo_3
    weights = np.full(shape = n, fill_value = 1/n)
    parameters = np.array([Z_1, np.full(shape = n, fill_value = h)])
    
    return weights, parameters