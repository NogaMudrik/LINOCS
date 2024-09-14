# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:31:28 2024


"""


from main_functions_lookahead_linocs import *
# Generate synthetic data for a linear dynamical system

    
    
    
    
    
def train_DAgger(seed=0, T=500, l2_w=0, weights_decay=1.01, num_iterations=100, noise_std=0.1, A_true=[], B_true=[], obs_data=[], dim=[], A_init = [], B_init = [],
                 return_full = True, A_init_type = 'step', DAD  = False):
    if  weights_decay != 1:
        raise ValueError(' weights_decay is not 1. future implementation')
    np.random.seed(seed)
    if checkEmptyList(obs_data):
        if checkEmptyList(A_true) or checkEmptyList(B_true):
            if checkEmptyList(dim) or  checkEmptyList(T):
                raise ValueError('if not provideing data nor A/B you must provide dimension and time!')
            else:
                print('creates random data')
                
                A_true = create_rotation_mat(theta = 0.05) # True dynamics matrix
                B_true = np.random.rand(dim,1)  # True input matrix

            x_true = np.random.rand(dim, T)
            u = np.random.randn(1, T)  # Input signal
       
            
        for t in range(1, T):
            x_true[:, t] = (np.dot(A_true, x_true[:, t - 1]).reshape((-1,1)) + B_true.reshape((-1,1))).flatten()
        obs_noise = noise_std * np.random.randn(dim, T)
        obs_data = x_true + obs_noise
    else:
        dim = obs_data.shape[0]
    if 'dim' not in locals():
        dim = obs_data.shape[0]

    
    if checkEmptyList(A_init):
        # Define the Dagger algorithm
        if A_init_type == 'step':
            Bs, A_init = find_Bs_for_dynamics(obs_data, 1, w_offset = True)

            Bs = Bs[0]
            A_init = A_init[0]
        elif A_init_type == 'rand':
            A_init = np.random.randn(dim, dim)
        else:
            raise ValueError('undefined A_init_type')
    if checkEmptyList(B_init):
        if A_init_type == 'step':
            B_init = Bs[:,-1].reshape((-1,1))
        elif A_init_type == 'rand':            
            B_init = np.random.randn(dim, 1)
        else:
            raise ValueError('undefined A_init_type')
            
    A_hat = A_init.copy()
    B_hat = B_init.copy()
    
    for _ in range(num_iterations):
        # Forward pass
        x_pred = np.zeros_like(obs_data)
        for t in range(1, T):
            x_pred[:, t] = (np.dot(A_hat, x_pred[:, t - 1]).reshape((-1,1)) +  B_hat.reshape((-1,1))).flatten()
    
        if not DAD:
            # Labeling using observed data
            labeled_data = np.hstack([obs_data[:, 1:], x_pred[:, 1:]])
        
            # Update parameters        
            combined_data = np.hstack([obs_data[:,:-1], x_pred[:,:-1]])
            
        else:            # Labeling using observed data
            labeled_data = obs_data[:, 1:]
        
            # Update parameters        
            combined_data = x_pred[:,:-1]
    
    
        combined_data = np.vstack([combined_data, np.ones((1, combined_data.shape[1]))])    
        if l2_w == 0:
            params = labeled_data @ np.linalg.pinv(combined_data)
        else:
            labeled_data = np.hstack([labeled_data, np.zeros((labeled_data.shape[0],combined_data.shape[0]))])
            combined_data = np.hstack([combined_data, l2_w*np.eye(combined_data.shape[0])])
            params = labeled_data @ np.linalg.pinv(combined_data)
        
        A_hat = params[:,:-1].reshape(dim, dim)
        B_hat = params[:,-1].reshape(-1, 1)
        if return_full:
            if 'x_true' not in locals():
                x_true = []
            return A_hat, B_hat, A_true, B_true, x_true, obs_data, A_init, B_init
    return A_hat, B_hat
    
    

# example run:
#    A_hat, B_hat, A_true, B_true, x_true, obs_data, A_init, B_init = train_DAgger(T= 100, dim =3, DAD = True)