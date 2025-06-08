# Libraries
import numpy as np
from sklearn.mixture import GaussianMixture

# EM Algorithm
def EM_Algorithm(Z_1, n_components = "BIC", n_components_trials = 15):
    """
    Fits a Gaussian Mixture Model using the EM algorithm.

    Parameters:
    - Z_1 (np.array): Sample array.
    - n_components (string or int): How many components are considered in the EM Algorithm. 
    If string, only admitted "BIC". If int, it is assigned that value.
    - n_components_trials (int): How many components are tried if n_components is "BIC".
    
    Return:
    - weights (np.array): Weights of each distribution in the mixture.
    - parameters (np.array): Parameters of each distribution in the mixture. It is 2-dimensional array. 
    The first dimension is the mean, the second one is the standard deviation.
    
    """  
    Z_1 = Z_1.reshape(-1, 1)
    if isinstance(n_components, str):
        if n_components == "BIC":
            lowest_bic = np.inf
            bic_scores = []
            best_gmm = None
            n_components_range = range(1, n_components_trials)
            
            for k in n_components_range:
                gmm = GaussianMixture(n_components = k, covariance_type = 'full', random_state = 123)
                gmm.fit(Z_1)
                bic = gmm.bic(Z_1)
                bic_scores.append(bic)
            
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    
            # Get best model parameters
            best_n_components = n_components_range[np.argmin(bic_scores)]
            weights = best_gmm.weights_
            means = best_gmm.means_.flatten()
            standard_deviations = np.sqrt(np.array([np.diag(cov) for cov in best_gmm.covariances_]).flatten())
            parameters = np.row_stack((means, standard_deviations))
            print(f"The lowest BIC was achieved with {best_n_components} components." )
            
        else:
            raise ValueError("If n_components is a string, it must be 'BIC'")
        
    elif isinstance(n_components, int):
        gmm = GaussianMixture(n_components = n_components, covariance_type = 'full', random_state = 123)
        gmm.fit(Z_1)

        # Get best model parameters
        weights = gmm.weights_
        means = gmm.means_.flatten()
        standard_deviations = np.sqrt(np.array([np.diag(cov) for cov in gmm.covariances_]).flatten())
        parameters = np.row_stack((means, standard_deviations))
        
    else:
        raise ValueError("bw_method must be a string or int.")
    
    return weights, parameters


# KDE
def KDE(Z_1, bw_method = "Silverman"):
    r"""
    
    Kernel density estimation for the pinning point distribution.
    
    Parameters:
    - Z_1 (np.array): Sample array.
    - bw_method (string or float): How bandwith is calculated. 
    If string, only admitted "Silverman" and "Scott". If float, it is assigned that value.
    
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
    unique_vals, inverse_indices, counts = np.unique(Z_1, return_inverse=True, return_counts=True)
    weights = counts/sum(counts)
    parameters = np.array([unique_vals, np.full(shape = len(unique_vals), fill_value = h)])
    
    return weights, parameters