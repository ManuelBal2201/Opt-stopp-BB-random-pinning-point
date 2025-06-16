# Libraries
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
    
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

# Log-likelihood of sigma
def sigma_log_likelihood(sigma, tilde_z_t, tilde_t, tilde_m, tilde_gamma2, pi, T):
    n = len(tilde_t)
    k = len(tilde_m)
    log_likelihood = 0

    for i in range(n - 1):
        ti = tilde_t[i] / T
        tip1 = tilde_t[i+1] / T
        tilde_z = tilde_z_t[i]
        z_ti = (tilde_z_t[i] - tilde_z) / (sigma * np.sqrt(T))

        inner_sum = 0
        for j in range(k):
            m_j = (tilde_m[j] - tilde_z) / (sigma * np.sqrt(T))
            gamma_j2 = tilde_gamma2[j] / (sigma**2 * T)

            A = ti / (2 * (1 - ti)) + 1 / (2 * gamma_j2)
            B = z_ti / (1 - ti) + m_j / gamma_j2
            C = (B**2) / (4 * A) - (m_j**2) / (2 * gamma_j2)

            denom = 0
            for h in range(k):
                mh = (tilde_m[h] - tilde_z) / (sigma * np.sqrt(T))
                gamma_h2 = tilde_gamma2[h] / (sigma**2 * T)
                Ah = ti / (2 * (1 - ti)) + 1 / (2 * gamma_h2)
                Bh = z_ti / (1 - ti) + mh / gamma_h2
                Ch = (Bh**2) / (4 * Ah) - (mh**2) / (2 * gamma_h2)
                denom += pi[h] * np.exp(Ch) * np.sqrt(np.pi / Ah)

            w_ij = pi[j] * np.exp(C) * np.sqrt(np.pi / A) / denom
            m_j_ti = B / (2 * A)
            gamma_j_ti2 = 1 / (2 * A)

            a_i = (tip1 - ti) / (1 - ti)
            b_i = (1 - a_i) * z_ti
            lambda2_i = (1 - tip1) * (tip1 - ti) / (1 - ti)
            hat_lambda_i = np.sqrt(lambda2_i) / a_i
            sigma2_ij = a_i**2 * (hat_lambda_i**2 + gamma_j_ti2)

            z_tip1 = (tilde_z_t[i+1] - tilde_z) / (sigma * np.sqrt(T))
            arg = z_tip1 - b_i - a_i * m_j_ti
            phi_val = norm.pdf(arg, scale=np.sqrt(sigma2_ij))
            inner_sum += w_ij * phi_val

        log_likelihood += np.log(inner_sum)

    return log_likelihood