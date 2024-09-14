# -*- coding: utf-8 -*-

def snythetic_evaluation(full_A, full_phi, real_full_A = {}, real_full_phi = {}):
    #     given ground truth  dynamics - compare the results
    #d_full = fnp.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
    if len(real_full_A ) == 0 or len(real_full_phi) == 0:
        d_full = np.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
        real_full_A = d_full['A']
        real_full_phi = d_full['phi']
        
    ordered_A = []
    ordered_phi = []
    for counter, (cond, phi1) in enumerate(real_full_phi.items()):     
        phi2, A, _ = match_times(phi1, full_phi[:,:,counter].T, full_A[:,:,counter]) #for counter, (cond, phi1) in enumerate(real_full_phi)
        ordered_phi.append(phi2)
        ordered_A.append(A)
    full_phi_ordered = np.dstack(ordered_phi)
    full_A_ordered = np.dstack(ordered_A)
    
    return full_phi_ordered, full_A_ordered

def run_existing_methods(data, p, methods_to_compare = ['adam_svd','hosvd','parafac','tucker', 'HOOI', 'adam_pca'],
                         params_parafac = {}, params_tucker = {}):
    # the mathods are taken from http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    # user guide http://tensorly.org/stable/user_guide/quickstart.html#tensor-decomposition
    results = {}
    
    if 'adam' in methods_to_compare:
        A_adam, phi_adam = run_adam_svd(data, p)
        results['adam'] = {'A':A_adam, 'phi':phi_adam}
    if 'adam_pca_global' in methods_to_compare:
        A_adam, phi_adam = run_adam_pca(data, p)
        results['adam_pca_global'] = {'A':A_adam, 'phi':phi_adam}
    if 'adam_pca_local' in methods_to_compare:
        A_adam, phi_adam = run_adam_pca_local(data, p)
        results['adam_pca_local'] = {'A':A_adam, 'phi':phi_adam}
        
    if 'tucker' in methods_to_compare:
        A_tucker, phi_tucker, _,_ = run_tucker(data, p = p, params_tucker = params_tucker)
        results['tucker'] = {'A':A_tucker, 'phi':phi_tucker}
    if 'parafac' in methods_to_compare:
        A_parafac, phi_parafac, _ = run_parafac(data, p = p, params_parafac = params_parafac)
        results['prafac'] = {'A':A_parafac, 'phi':phi_parafac}
    return results
    
    
def run_adam_svd(data, p = 10, max_TK = 1000):
    print('dat')
    print('MovToD2(data)')
    dd = MovToD2(data)
    #A, s, VT = np.linalg.svd(MovToD2(data), full_matrices = False)
    A, s, VT = np.linalg.svd(dd)

    A = A[:p]
    print('VT shape')
    print(VT.shape)
    
    #phi = split_stacked_data(VT, T = data.shape[1], k = data.shape[2]).transpose((1,0,2))
    phi = VT
    return A, phi
    
from sklearn.decomposition import PCA    
def run_adam_pca(data, p = 10, max_TK = 1000):
    # Create a PCA instance and specify the number of components you want to extract
    if isinstance(data, list):
        datap = np.hstack([data_i for data_i in data])
    elif len(data.shape) == 3:
        datap = np.hstack([data[:,:,i] for i in range(data.shape[2])])
    else:
        datap = data
    num_components = p
    pca = PCA(n_components=num_components)
    
    # Fit the PCA model to your data
    A = pca.fit_transform(datap)
    
    # Get the principal components (components matrix) from the fitted PCA model
    phi = pca.components_
    
    return A, phi    
    
def run_adam_pca_local(data, p = 10, max_TK = 1000):
    # Create a PCA instance and specify the number of components you want to extract
    #     if isinstance(data, list):
    #         datap = np.hstack([data_i for data_i in data])
    #     elif len(data.shape) == 3:
    #         datap = np.hstack([data[:,:,i] for i in range(data.shape[2])])
    #     else:
    #         datap = data
    A = []
    phi = []
    num_components = p    
    for i in range(data.shape[2]):
        datap = data[:,:,i]
        pca = PCA(n_components=num_components)

        # Fit the PCA model to your data
        A_l = pca.fit_transform(datap)
        A.append(A_l)
        # Get the principal components (components matrix) from the fitted PCA model
        phi_l = pca.components_
        phi.append(phi_l)
    return A, phi   


def run_tucker(data, p = 10, params_tucker = {}):
    """
    explanation: factors[0] is A, 
    ignoring core?! :( 
    explanation: http://tensorly.org/stable/modules/generated/tucker-function.html#tensorly.decomposition.tucker

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 10.
    params_tucker : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
     : TYPE
        DESCRIPTION.
    core : TYPE
        DESCRIPTION.
    factors : TYPE
        DESCRIPTION.

    """
    k = data.shape[2]
    T = data.shape[1]
    core, factors = tucker(data, rank=[p, p, T], ** params_tucker)
    A_tucker = factors[0]
    phi_base_tucker = factors[1]
    k_tucker = factors[2]
    
    print('k_tucker.hape')
    print(k_tucker.shape)
    print('phi_base_tucker')
    print(phi_base_tucker.shape)
    phi_tucker = np.dstack([phi_base_tucker*k_tucker[k_spec, k_spec] for k_spec in range(k)])
    
    
    return A_tucker, phi_tucker, core, factors
    
    
    
def run_parafac(data, p = 10, params_parafac = {}):   
    """
    decomposition: http://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html#tensorly.decomposition.parafac
    parafac paper: https://www.cs.cmu.edu/~pmuthuku/mlsp_page/lectures/Parafac.pdf
    """
    N = data.shape[0]
    T = data.shape[1]
    k = data.shape[2]
    # gives me a list of N x p; T x p; k x p
    #np.random.rand(N,T,k)
    factors = parafac(data, rank=p, **params_parafac)
    factors_f = factors.factors
    factors_w = factors.weights
    
    full_A = []
    full_phi = []
    
    A_parafac = factors_f[0]
    phi_base_parafac = factors_f[1]
    print('phi_base_parafac')
    print(phi_base_parafac.shape)
    k_parafac = factors_f[2]
    for k_spec in range(k):         
        phi_parfac = np.dstack([phi_base_parafac*k_parafac[k_spec,:].reshape((1,-1)) for k_spec in range(k)] )
        
    return A_parafac, phi_parfac, factors
    
    
    
    