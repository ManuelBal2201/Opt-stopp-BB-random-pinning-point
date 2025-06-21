# Libraries
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
    
# EM Algorithm
def EM_Algorithm(Z_1, n_components = "BIC", n_components_trials = 15):
    """
    Fits Z_1 to a Gaussian mixture using the EM algorithm.

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
        if n_components == "BIC": # Number of components chosen using BIC
            # Initialise variables
            lowest_bic = np.inf
            bic_scores = []
            best_gmm = None
            n_components_range = range(1, n_components_trials)
            
            # Apply EM algorithm from 1 Gaussian mixture to n_components_range Gaussian mixture
            for k in n_components_range:
                # Apply EM algorithm
                gmm = GaussianMixture(n_components = k, covariance_type = 'full', random_state = 123)
                gmm.fit(Z_1)
                
                # Compute BIC
                bic = gmm.bic(Z_1)
                bic_scores.append(bic)
                
                # Update the best Gaussian mixture if the current BIC is lower than the lowest previously computed
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    
            # Get best model parameters
            best_n_components = n_components_range[np.argmin(bic_scores)] # Optimal number of components
            weights = best_gmm.weights_ # Weights
            means = best_gmm.means_.flatten() # Means
            standard_deviations = np.sqrt(np.array([np.diag(cov) for cov in best_gmm.covariances_]).flatten()) # Standard deviation
            parameters = np.row_stack((means, standard_deviations)) # Create parameters np.array to be directly introduced in optimal_stopping_montecarlo_3
            print(f"The lowest BIC was achieved with {best_n_components} components." ) # Print the optimal number of components
            
        else:
            raise ValueError("If n_components is a string, it must be 'BIC'")
        
    elif isinstance(n_components, int): # Number of components chosen specifically
        # Apply EM Algorithm
        gmm = GaussianMixture(n_components = n_components, covariance_type = 'full', random_state = 123)
        gmm.fit(Z_1)

        # Get best model parameters
        weights = gmm.weights_ # Weights
        means = gmm.means_.flatten() # Means
        standard_deviations = np.sqrt(np.array([np.diag(cov) for cov in gmm.covariances_]).flatten()) # Standard deviations
        parameters = np.row_stack((means, standard_deviations)) # Create parameters np.array to be directly introduced in optimal_stopping_montecarlo_3
        
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
    # Initialise variables
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
    
    # Compute the weights and parameters of the Gaussian mixture. They are thought to be directly introduced in optimal_stopping_montecarlo_3
    unique_vals, inverse_indices, counts = np.unique(Z_1, return_inverse=True, return_counts=True)
    weights = counts/sum(counts) # Weights
    parameters = np.array([unique_vals, np.full(shape = len(unique_vals), fill_value = h)]) # Create parameters np.array to be directly introduced in optimal_stopping_montecarlo_3
    
    return weights, parameters

# Log-likelihood of sigma
def sigma_log_likelihood_vectorized(volatility, tilde_z_t, tilde_t, tilde_m, tilde_gamma, weights):
    """
    
    Volatility log-likelihood function value given all its variables.
    
    Parameters:
    - volatility (float): Volatility of the process.
    - tilde_z_t (np.array): Observations of the process.
    - tilde_t (np.array): Times of the observations of the process.
    - tilde_m (np.array): Means of the terminal density Gaussian mixture.
    - tilde_gamma (np.array): Standard deviations of the terminal density Gaussian mixture.
    - weights (np.array): Weights of the terminal density Gaussian mixture.
    
    Return:
    - log-likelihood (float): Volatility log-likelihood function value given the inputs.
    
    """
    
    # Ensure all inputs are NumPy arrays to avoid Pandas broadcasting issues
    tilde_z_t = np.asarray(tilde_z_t)
    tilde_t = np.asarray(tilde_t)
    tilde_m = np.asarray(tilde_m)
    tilde_gamma = np.asarray(tilde_gamma)
    weights = np.asarray(weights)

    # Normalise time and values
    n = len(tilde_t)
    t = tilde_t
    z_t = (tilde_z_t - tilde_z_t[0]) / volatility
    m = (tilde_m - tilde_z_t[0]) / volatility
    gamma2 = (tilde_gamma ** 2) / (volatility ** 2)

    # Time steps and z-values at steps
    ti = t[:-1]
    ti1 = t[1:]
    zti = z_t[:-1]
    zti1 = z_t[1:]

    # Expand shapes for broadcasting: (n-1, 1) + (k,) => (n-1, k)
    A = ti[:, None] / (2 * (1 - ti[:, None])) + 1 / (2 * gamma2)
    B = zti[:, None] / (1 - ti[:, None]) + m / gamma2
    C = B**2 / (4 * A) - m**2 / (2 * gamma2)

    # Compute weights w_{i,j}
    safe_exp_C = np.exp(C - np.max(C, axis=1, keepdims=True))  # for stability
    weights_numerator = weights * safe_exp_C * np.sqrt(np.pi / A)
    weights_ty = weights_numerator / weights_numerator.sum(axis=1, keepdims=True)

    # m_{j, t_i, z_ti}, gamma_{j, t_i, z_ti}
    m_tz = B / (2 * A)
    gamma_tz2 = 1 / (2 * A)

    # a_i, b_i, lambda_hat_i^2
    a = (ti1 - ti) / (1 - ti)
    b = (1 - a) * zti
    lambda2 = (1 - ti1) * (ti1 - ti) / (1 - ti)
    lambda_hat2 = lambda2 / a**2

    # Compute means and stds for norm.pdf
    means = b[:, None] + a[:, None] * m_tz
    stds = np.sqrt(a[:, None]**2 * (lambda_hat2[:, None] + gamma_tz2))

    # Evaluate norm.pdf for each zti1[i] against each component j
    pdf_vals = norm.pdf(zti1[:, None], loc=means, scale=stds)

    # Weighted sum over j for each i
    inner_sums = np.sum(weights_ty * pdf_vals, axis=1)

    # Final log-likelihood sum
    log_likelihood_sum = np.sum(np.log(inner_sums))

    return log_likelihood_sum - (n - 1) * np.log(volatility)

# Process classificator
def process_classificator(N, value_function, X_vals, volatility, Z_0, pair_df, neighbours = 7):
    r"""
    
    Identify if each observation of the process is in the stopping or in the continuation region.
    
    Parameters:
    - N (int): Number of temporal steps.
    - value_function (np.array): Value function for all the spatial points on each time step.
    - X_vals (np.array): Spatial grid we are considering.
    - volatility (float): Volatility of the process.
    - Z_0 (float): First observation of the process.
    - pair_df (pd.dataframe): Process observations dataframe. The first column is the temporal point, and the second is the spatial.
    - neighbours (int): Neighbours considered in KNN to assign a region for the observation.
    
    Return:
    - pair_df (pd.dataframe): Input process observations dataframe with a third columns: the predicted region.
    
    """
    
    # Initialise variables
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
        
        
    else:
        # Adjust meshgrid
        T, X = np.meshgrid(t_mesh, X_vals)
        
        # Define colors and mapping
        comparison = value_function <= np.tile(X_vals, (N, 1)) # Boolean matrix
    
    # Scale the spatial grid
    X = X*volatility + Z_0

    # Flatten data and build DataFrame
    df = pd.DataFrame({
        'Temporal': T.flatten(),
        'Spatial': X.flatten(),
        'Comparison': comparison.T.flatten()
    })
    
    # Encode Comparison to 0/1 for KNN 
    le = LabelEncoder()
    df['Comparison_encoded'] = le.fit_transform(df['Comparison'])
    
    # Train KNN Classifier 
    X_train = df[['Temporal', 'Spatial']].values
    y_train = df['Comparison_encoded'].values
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    knn.fit(X_train, y_train)
    
    # Predict encoded labels
    pred_encoded_all = knn.predict(pair_df.values)
    
    # Decode predictions back to original labels
    pred_labels_all = le.inverse_transform(pred_encoded_all)
    
    # Attach predictions to the input DataFrame
    pair_df['Prediction'] = pred_labels_all

    
    return pair_df