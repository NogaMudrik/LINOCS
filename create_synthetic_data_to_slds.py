# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime as datetime2

cur_path = os.getcwd()
# Get the parent folder



parent_folder = os.path.dirname(cur_path)
if 'ALL_' not  in parent_folder:
    # Construct the path to the "november 2023" folder
    november_2023_path = os.path.join(parent_folder, "november 2023")

    # Change the current working directory to "november 2023"
    os.chdir(november_2023_path)
    from   main_functions_lookahead_linocs import *  # Assuming this is your new module name

    os.chdir(cur_path)

else:
    from   main_functions_lookahead_linocs import * 



def create_snyth_x_one_session(c,F,x,  seed = 0,  num_regions = 3):
    """
    Generate a synthetic session based on input parameters.
    
    Parameters:
    - c (array-like): Coefficient matrix.
    - F (array-like): Some other matrix used in the creation process.
    - x (array-like): Input array for the generation process.
    - seed (int, optional): Seed for random number generation. Defaults to 0.
    - num_regions (int, optional): Number of regions. Defaults to 3.
    
    Returns:
    array-like: The generated synthetic session.
    """
    c = c.copy()
    np.random.seed(seed)    
    x = np.random.rand(num_regions,1) 
    x_full = create_reco_new(x, c, F)     
    return x_full  

    
def create_periodic_sparse_c(M, T, value_insert = 1.11, period_min = 5, period_max = 20, seed = 0, making_sure_all = False):
    """
    Create a periodic sparse matrix.
    
    Parameters:
    - M (int): Number of rows in the matrix.
    - T (int): Number of columns in the matrix.
    - value_insert (float, optional): Value to insert in non-zero elements. Defaults to 1.11.
    - period_min (int, optional): Minimum period for creating periodic elements. Defaults to 5.
    - period_max (int, optional): Maximum period for creating periodic elements. Defaults to 20.
    - seed (int, optional): Seed for random number generation. Defaults to 0.
    
    Returns:
    array-like: The created periodic sparse matrix.
    """
    np.random.seed(seed +9)
    c = np.zeros((M,T))
    max_num_periods = int(np.ceil(T/period_min))
    periods = np.random.randint(period_min, period_max, size = max_num_periods)
    # below is which dynamics will be active?
    if making_sure_all:
        # firest make sure each is presented once
        keep_nonzero_init = np.arange(M)
        np.random.shuffle(keep_nonzero_init)
        if max_num_periods > M:
            keep_nonzero_second = np.random.randint(0, M, max_num_periods-M)     
            keep_nonzero = np.concatenate((keep_nonzero_init, keep_nonzero_second))
        else:
            keep_nonzero = keep_nonzero_init
        
    else:
        keep_nonzero = np.random.randint(0, M, max_num_periods)    
    nonzero_indices = np.repeat(keep_nonzero, periods)    
    c[list(nonzero_indices[:T]), list(np.arange(T))] = value_insert
    return c
    
    


def create_rotation_mat(theta = 0, axes = 'x', dims = 3):
    """
    Create a rotation matrix for a specified angle and axes.
    
    Args:
        theta (float, optional): The rotation angle in radians (default: 0).
        axes (str, optional): The axes of rotation:
                              - 'x': Rotation around the x-axis (default).
                              - 'y': Rotation around the y-axis.
                              - 'z': Rotation around the z-axis.
        dims (int, optional): The number of dimensions for the rotation matrix:
                              - 3: 3D rotation matrix (default).
                              - 2: 2D rotation matrix.
    
    Returns:
        numpy.ndarray: The rotation matrix.
    
    Raises:
        ValueError: If dims is not 2 or 3.
    
    """
    if dims == 3:
        if axes.lower() == 'x':
            rot_mat = np.array([[1,0,0],
                                [0,np.cos(theta), -np.sin(theta)], 
                                [0, np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],
                                [0,1, 0], 
                                [-np.sin(theta),0, np.cos(theta)]])
        elif  axes.lower() == 'z':
            rot_mat = np.array([[np.cos(theta),-np.sin(theta),0],
                                [np.sin(theta),np.cos(theta), 0], 
                                [0, 0, 1]])
    elif dims == 2:
        if axes.lower() == 'x':
            rot_mat = np.array([[0,np.cos(theta), -np.sin(theta)], 
                                [0, np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],                            
                                [-np.sin(theta),0, np.cos(theta)]])
        else:
            raise ValueError('axes is invalid')
        
    else: 
        raise ValueError('dims should be 2 or 3')
    return rot_mat    
    



    
def find_switching(data, M, wind = 15, K = 5, interval_start = 2, params = {}, to_plot = False, 
                   to_inference_A = True, weights = [], weights_style = 'dec', to_create_data = True, 
                   x = [], x_noisy = [], cc = [], cc_gauss = [], w_offset = False, to_inference_one = True,
                   compare  = True, to_avg = True, reoptimize = True, k_full = 8):
    # for full switch - > full_switch = False,  with_gauss = False
    params_new = {'theta': np.pi/5,
     'T': 100,
     'period_min':  30,
     'period_max': 40,
     'seed': 0,
     'x0': [],
     'noise_level': 0.2,
     'w_gauss': True
    }
    
    params_new = {**params_new, **params}
    
    """
    create data
    """
    if to_create_data: 
        print('creating data')
        x, cc, cc_gauss, F,  x_noisy = create_synth_for_lookahead_slds(**params_new)
        operators = find_local_operators(F, zs  = [], cs = cc_gauss)
        z_ground_truth =  c2z(cc_gauss)
    else:
        if checkEmptyList(x_noisy)  or checkEmptyList(x): # or checkEmptyList(cc) or checkEmptyList(cc_gauss):
            raise ValueError('you must provide vals for  x, x_noisy, cc, cc_gauss; unless to_create_data = True')
    
    
    if to_plot:
        fig, ax = create_3d_ax(1, 2, figsize = (20,5))
        plot_3d(x, ax = ax[0])
        plot_3d(x_noisy, ax = ax[1])
        
    """
    find weights
    """
    T = x_noisy.shape[1]
    if checkEmptyList(weights):
        print('weights')
        if weights_style == 'dec':
            weights = (np.arange(T) + 1)[::-1]
        elif weights_style == 'er':
            weights = []        
        else:
            raise ValueError('/')
    print('looking for A')    
    identified_As = []
    if  to_inference_A:        
        
        starting_points = np.arange(0, T-1, interval_start)        
        #ess = np.zeros(((len(starting_points),T)))*np.nan        
        best_sc_full = []
        identified_As = []
        
        for c, starting_point in enumerate(starting_points):
            print('starting: %d'%starting_point)
            k = np.min([K, wind])
            opt_A = train_linear_system_opt_A(x_noisy[:, starting_point:starting_point + wind], 
                                              k, w_offset = w_offset, weights = weights)#train_linear_system_opt_A(xs_small[:,0].reshape((-1,1)), k, w_offset=False, weights=[])
            #rec1_hat_opt = propagate_dyn_based_on_operator(x_noisy[:,starting_point],
            #                                               opt_A, 
            #                                               offset = [] , max_t = xs[:,starting_point: ].shape[1] - 1 )#offset_hat_optxs_small[:,starting_point:].shape[1] -1
            identified_As.append(opt_A)
            #es = ((rec1_hat_opt - xs[:,starting_point: ])**2).mean(0)
        


    """
    find one step reco
    """
    identified_As_ones = []
    if  to_inference_one:     
        for c, starting_point in enumerate(starting_points):
            #print('starting: %d'%starting_point)
            k = np.min([k_full, wind])
            A_noise = train_linear_system_opt_A(x_noisy[:, starting_point:starting_point + interval_start], 1, w_offset=False, weights=weights[:2])#train_linear_system_opt_A(xs_small[:,0].reshape((-1,1)), k, w_offset=False, weights=[])
            rec1_noise = propagate_dyn_based_on_operator(x_noisy[:,starting_point],opt_A, offset = [] , max_t = x_noisy[:,starting_point: ].shape[1] - 1 )#offset_hat_optxs_small[:,starting_point:].shape[1] -1
            identified_As_ones.append(A_noise)
            #es = ((rec1_hat_opt - xs[:,starting_point: ])**2).mean(0)
            
    """
    compare results and find basis operators
    """
    if compare: 
        print('compare')
        starting_points_shift = starting_points + int(wind/2)
        fig, axs_full = plt.subplots(2,5, figsize = (40,15), sharex = 'col', sharey = 'col')
        params_plot_in = {'heat1': {'cbar':False, 'vmin': 0, 'vmax':0.3, 'square':True},
                         'heat2':{'cbar':False, 'vmin': 0, 'vmax':1, 'square':True},
                         'labels1': {'lw': lw, 'color':colors['LINOCS']},
                         'labels2': {'lw': lw, 'color':colors['LINOCS']}}
    
        
        params_plot_in = {'heat1': {'cbar':False, 'vmin': 0, 'vmax':0.3, 'square':True},
                         'heat2':{'cbar':False, 'vmin': 0, 'vmax':1, 'square':True},
                         'labels1': {'lw': lw, 'color':colors['noisy']},
                         'labels2': {'lw': lw, 'color':colors['noisy']}}
        labels, As, As_dstack, As_dstack_reorder, mat_er, corr_mat = find_A_basis_from_operators(identified_As, reorder_ref=F, params_plot = params_plot_in, axs = axs_full[0], percentile=False, thres_percentile = 0.1/5, fig = fig, starting_points = starting_points_shift, wind_avg =7, thres_corr_soft=0.6)
        labels_ones, As_ones, As_dstack_ones, As_dstack_ones_reorder, mat_er_one, corr_mat_one = find_A_basis_from_operators(identified_As_ones, reorder_ref=F, params_plot = params_plot_in,percentile=False, thres_percentile = 0.05, fig = fig, starting_points = starting_points_shift, wind_avg  = 3, thres_corr_soft=0.1 ,axs = axs_full[1])

    if to_plot:
        

    
        
        #labels_ones, As_ones, As_dstack_ones, As_dstack_ones_reorder = find_A_basis_from_operators(identified_As_ones, reorder_ref=F, params_plot = params_plot_in, axs = axs_full[1], percentile=False, thres_percentile = 0.05, fig = fig, starting_points = starting_points_shift, wind_avg  = 7)
        
        
        #[add_labels(axs_full[0,i], title = title , ylabel = 'LINOCS' if i == 0, xlabel = 'Time' ) for i, title in ['A diff', 'A_diff thres', 'labels before thres', 'labels_after thres']]
        [add_labels(axs_full[0,i], title = title , ylabel = 'LINOCS' if i == 0 else '', xlabel = 'Time', zlabel = '', xlabel_params=label_params,ylabel_params=label_params,title_params=label_params ) for i, title in enumerate(['A diff', 'A_diff thres', 'labels before thres', 'labels_after thres', 'labels_after avg'])]
        [add_labels(axs_full[1,i], title = title , ylabel = '1 step' if i == 0 else '', xlabel = 'Time' , zlabel = '',xlabel_params=label_params,ylabel_params=label_params,title_params=label_params) for i, title in enumerate(['A diff', 'A_diff thres', 'labels before thres', 'labels_after thres'])]
        
        [ax.set_ylim(bottom = 0) for ax in axs_full[:,2:].flatten()]
        [ax.plot(cc_gauss.T, color ='gray') for ax in axs_full[:,2:].flatten()]
        
        
        fig.tight_layout()
        
        plt.savefig(path_save + os.sep + 'find_periods_%s.png'%addi, transparent = True,bbox_inches='tight')
        plt.savefig(path_save + os.sep +  'find_periods_%s.svg'%addi, transparent = True,bbox_inches='tight')    
                        
    As_opt = [As_dstack_reorder[:,:,i] for i in range(As_dstack_reorder.shape[2])]
            
    T = x.shape[1]
    start_end, inverse_start_end  =  find_start_end_points(labels, starting_points, T)
    interp_func = interp1d(starting_points
                           , labels, kind='linear', fill_value='extrapolate')
    
    # Define the new time range (T > m)
    T_ser = np.arange(T) #np.linspace(times.min(), times.max() + 1, num=desired_number_of_points)
    
    # Perform interpolation
    interpolated_labels = interp_func(T_ser)
    

    c_final = np.zeros((len(F), x.shape[1]))

    for count in range(np.max([start_end.shape[0], inverse_start_end.shape[0]])):
        if inverse_start_end.shape[0] > count:
            #start_end_i = start_end[count]        
            start_end_constant = inverse_start_end[count]    
            start = start_end_constant[0]
            end = start_end_constant[1]
            z_val = int(np.round(np.nanmean(interpolated_labels[start:end])))
            
            c_final[z_val - 1, np.max([0,start-1]) : np.min([end, T+1])] = 1 #put_z #cs_dstack[:,:,0]
    
        
        if start_end.shape[0] > count:
            start_end_i = start_end[count]    
            A_cur_rewighs, cs_rewighs, cs_dstack, As_stores, weights, stores_reco, es, reco_look = find_A_decomposed_K_steps(x_noisy[:,start_end_i[0]:start_end_i[1] +4 ], As_opt, K = 1, sigma1 = 0.2, sigma2 = 0.4, smooth_w=2.9, l1_w = 1.1, c_start = c_final[:,start_end_i[0]])
            c_final[:, start_end_i[0] +1 :start_end_i[1]-1+4] = cs_dstack[:,1:,0]
            
    if to_avg:                    
        c_final = mov_avg(c_final, axis = 1, wind =7)
    if to_reopt:  
        print('re-opt')
        sols_c = []
        former = []
        reco_so_far = []
        for t in range(T-1):
            x_noisy_t = x_noisy[:,t]
            x_next = x_noisy[:,t+1]
            c_t_know = c_final_mov[:,t]
            sol = minimize(try_optimize, c_final_mov[:,0], method = 'BFGS', args = (As_opt, x_noisy_t, x_next, c_t_know, former,reco_so_far,t))
            #print(sol.x)
            sols_c.append(sol.x.reshape((-1,1)))
            former = sol.x
            reco_so_far = create_reco_new(x_noisy[:,0], np.hstack(sols_c),  As_opt, type_reco='lookahead')[:,-1]
        sols_c = np.hstack(sols_c).T    
    
        if to_plot:
            sns.heatmap(sols_c)
            plt.figure()
            sns.heatmap(np.vstack(cc_gauss).T)
        
            plt.figure()
            sns.heatmap(np.vstack(cc_gauss).T- sols_c)
            plt.figure()
            sns.heatmap(x_noisy, vmin = 0, vmax = 1) 
            plt.figure()
            sns.heatmap(x, vmin = 0, vmax = 1) 
            plt.figure()
            sns.heatmap(create_reco_new(x, cc_gauss,  As_opt, type_reco='lookahead'), vmin = 0, vmax = 1) 
            plt.figure()
            sns.heatmap(create_reco_new(x,sols_c.T,  As_opt, type_reco='lookahead'), vmin =0, vmax =1)
            plt.figure()
            sns.heatmap(create_reco_new(x,sols_c.T ,  As_opt, type_reco='lookahead')- x, vmin =0, vmax =1)    
        c_final = sols_c
    return c_final, As_opt, labels, As, As_dstack, As_dstack_reorder, labels_ones, As_ones, As_dstack_ones, As_dstack_ones_reorder, F, cc, cc_gauss, x, x_noisy, mat_er_one, corr_mat_one, mat_er_one, corr_mat_one


            
                
            
            
def dLDS_synth_create_Fs(num_Fs = 3, p = 3, theta = 1):
    """
    Generate a list of rotation matrices for a dynamic Linear Dynamical System (dLDS).
    
    Parameters:
    - num_Fs (int, optional): Number of rotation matrices to generate. Defaults to 3.
    - p (int, optional): Dimensionality of the rotation matrices. Defaults to 3.
    - theta (float, optional): Rotation angle in radians. Defaults to 1.
    
    Returns:
    list of array-like: A list containing rotation matrices for the specified number of dimensions.
    """
    #if not discrete:
    return  [create_rotation_mat(theta = theta, axes = axis, dims = p) for axis in ['x','y','z']]

  
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='Z', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], 
               ylabel_params = {'fontsize':19},zlabel_params = {'fontsize':19}, xlabel_params = {'fontsize':19}, 
               title_params = {'fontsize':19}, format_xticks = 0, format_yticks = 0):
  """
  This function add labels, titles, limits, etc. to figures;
  Inputs:
      ax      = the subplot to edit
      xlabel  = xlabel
      ylabel  = ylabel
      zlabel  = zlabel (if the figure is 2d please define zlabel = None)
      etc.
  """
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**zlabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend) > 0:  ax.legend(legend)
  if format_xticks > 0:
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_xticks))
  if format_yticks > 0:      
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_yticks))
      



def checkEmptyList(obj):
    """
    Check if the given object is an empty list.

    Args:
        obj (object): Object to be checked.

    Returns:
        bool: True if the object is an empty list, False otherwise.

    """    
    return isinstance(obj, list) and len(obj) == 0

def check_1d(mat):        
    return np.max(mat.shape) == len(mat.flatten())

def plot_3d(mat, ax = [], fig = []):
    if checkEmptyList(ax):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    if isinstance(mat, list):
        for mat_i in mat:
            ax.plot(mat_i[0], mat_i[1], mat_i[2])
    else:

        ax.plot(mat[0], mat[1], mat[2])
    
def plot_3d(mat, ax = [], fig = []):
    if checkEmptyList(ax):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    if isinstance(mat, list):
        for mat_i in mat:
            ax.plot(mat_i[0], mat_i[1], mat_i[2])
    else:
        ax.plot(mat[0], mat[1], mat[2])


# def create_reco_new(x,  coefficients, F, type_reco = 'lookahead', plus_one = 0,
#                     thres_max = 40, thres_min = 5.5,  seed = 0):
#     if checkEmptyList(x) and type_reco == 'lookahead':
#         np.random.seed(seed)
#         x = np.random.rand(F[0].shape[0]).reshape((-1,1))        
        
#     T = coefficients.shape[1] + 1
#     if not check_1d(x) and type_reco == 'lookahead':
#         x = x[:,0]
#         x = np.array(x).reshape((-1,1))
#     elif  type_reco == 'lookahead':
        
#         x = np.array(x).reshape((-1,1))
        
#     #t = 0
#     if type_reco == 'lookahead':
#         for t in range(T-1):
#             x = np.hstack([x,   (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @  x[:,-1] ).reshape((-1,1)) ])
        
        
#     else:
#         print(len(F))
#         x_hat = np.hstack([
#             (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @   x[:,t].reshape((-1,1))).reshape((-1,1))
#             for t in range(x.shape[1]-1)])
#         x = np.hstack([x[:,0].reshape((-1,1)), x_hat]) 
        
#     return x  
    
        
    
def create_synth_for_lookahead_slds(theta = 0.2, T = 100, period_min = 30, period_max = 40, seed = 0, x0 = [], noise_level = 0, 
                                    w_gauss = True, offset = [], making_sure_all = False, wind_gauss = 10):
    # this is the main fuile to run
    F = dLDS_synth_create_Fs(3, 3, theta)
    c = create_periodic_sparse_c(len(F), T + 30, value_insert = 1, period_min = period_min, period_max = period_max, 
                                 seed = seed, making_sure_all = making_sure_all )
    
    if w_gauss:
        c_gauss = gaussian_convolve(c, wind = wind_gauss)
        c_gauss = c_gauss/np.sum(c_gauss,0)
    else:
        c_gauss = c
    x = create_reco_new(x0, c_gauss, F, 'lookahead', offset = offset)
    #c_noisy = c_gauss + np.random.rand(*c.shape)*noise_level
    x_noisy = x  + np.random.randn(*x.shape)*noise_level #create_reco_new(x0, c_noisy, F, 'lookahead')
    return x, c, c_gauss, F,  x_noisy
    

def pad_mat(mat, pad_val, size_each = 1, axis = 1):
    if axis == 1:
        each_pad = np.ones((mat.shape[0], size_each))*pad_val
        mat = np.hstack([each_pad, mat, each_pad])
    else:
        each_pad = np.ones((size_each, mat.shape[1]))*pad_val
        mat = np.vstack([each_pad, mat, each_pad])        
    return mat

    

def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1, norm_sum = True, plot_gaussian = False):
    """
    Convolve a 2D matrix with a Gaussian kernel along the specified direction.
    
    Parameters:
        mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
        wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
        direction (int, optional): The direction of convolution. 
            1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.
    
    Returns:
        numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.
        
    Raises:
        ValueError: If 'direction' is not 0 or 1.
    """
    if direction == 1:
        gaussian = gaussian_array(2*wind,sigma)
        if norm_sum:
            gaussian = gaussian / np.sum(gaussian)
        if plot_gaussian:
            plt.figure(); plt.plot(gaussian)
        mat_shape = mat.shape[1]
        T_or = mat.shape[1]
        mat = pad_mat(mat, np.nan, wind)
        return np.vstack( [[ np.nansum(mat[row, t:t+2*wind]*gaussian)                    
                     for t in range(T_or)] 
                   for row in range(mat.shape[0])])
    elif direction == 0:
        return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
    else:
        raise ValueError('invalid direction')
    
    
def mov_avg(c, axis = 1, wind = 5):
    if len(c.shape) == 2 and axis == 1:
        return np.hstack([np.mean( c[:,np.max([i-wind, 1]):np.min([i+wind, c.shape[1]])],1).reshape((-1,1))
              for i in range(c.shape[1])])
    elif len(c.shape) == 2 and axis == 0:
        return mov_avg(c.T, axis = 1).T
    elif len(c.shape) == 3: # and axis == 0:
        return np.dstack([mov_avg(c[:,:,t], axis = axis) for t in range(c.shape[2])  ])
    else:
        raise ValueError('how did you arrive here? data dim is %s'%str(c.shape))
    
    

def gaussian_array(length,sigma = 1  ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
    
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    normalized_gaussian = gaussian / np.max(gaussian)
    return normalized_gaussian
    
    

                 


    

def create_synth_data_one_session(seed = 0, M = 3, T = 100, sigma = 0.8, 
                                  wind = 25, num_regions =3 , min_per_region = 3, max_per_region = 8,
                                  num_ens_per_region = 1, return_F = True, F = [],
                                  c_convolve = [], x_full = [], 
                                  period_min = 43, period_max = 50, std_noise = 0.05, value_insert = 1,
                                  w_noise = False, theta = 0.2,making_sure_all = False):
    """
    Generate synthetic data for a single session.
    
    Parameters:
    seed (int): Random seed for reproducibility.
    M (int): Number of factors.
    T (int): Number of time steps.
    sigma (float): Standard deviation for synthetic noise.
    wind (int): Window size for Gaussian convolution.
    num_regions (int): Number of regions.
    min_per_region (int): Minimum number of elements per region.
    max_per_region (int): Maximum number of elements per region.
    
    Returns:
    tuple: A tuple containing generated data.
        - c_convolve (ndarray): Convolved and thresholded factor matrix.
        - F (ndarray): Factor loading matrix.
        - x_full (ndarray): Latent state sequence.
        - y (ndarray): Observed data sequence.
        - D (ndarray): Mixing matrix.
    old: min_per_region = 3, max_per_region = 8,        
    """
    np.random.seed(seed)
    if not return_F and checkEmptyList(F):
        raise ValueError('you must provide F or calculate F')
        
    # create D
    num_per_region = create_snyth_num_per_region(min_per_region, max_per_region, num_regions = num_regions, seed = seed)

    D, D_mask = create_snyth_D(num_per_region, num_regions, num_ens_per_region, seed = seed)
    zero_cols = np.sum(D_mask,0) == 0
    D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-19)
    D[:,zero_cols] = 0
    # D_det = np.linalg.det(D)
    
    if return_F:
        # create F
        F = dLDS_synth_create_Fs(num_Fs = M, theta = theta)    
        # create c
    c = create_periodic_sparse_c(M, T, value_insert = value_insert, period_min = period_min, 
                                 period_max = period_max, seed = seed )
    #plt.figure(); plt.plot(c.T)
    c_convolve = gaussian_convolve(c, wind = wind, sigma = sigma, norm_sum = True, plot_gaussian = True)

    # create x
    #np.random.seed(0)
    x = np.random.rand(3,1)
    if w_noise:     
        cs_noisy = c_convolve + np.random.randn(*c_convolve.shape)*std_noise
    else:
        cs_noisy = c_convolve
    #plt.figure(); plt.plot(cs_noisy.T)
    cs_noisy = cs_noisy[:,15:]
    #plt.figure(); plt.plot(cs_noisy.T)
    
  
    
  
    
  
    
    #create_snyth_x_one_session(cs_noisy, F, x,  num_regions = num_regions )
    #plt.figure(); plt.plot(cs_noisy.T)
    # create y
    y = D @ x_full    
    
    return c_convolve, F, x_full, y, D, D_mask, cs_noisy, num_per_region
    
def find_local_operators(As, zs  = [], cs = []):
  """
  Find local operators based on input matrices and indices.

  Parameters:
  - As (list of array-like): List of matrices to be combined into local operators.
  - zs (list of int, optional): List of indices to select matrices from `As`. Defaults to an empty list.
  - cs (list of array-like, optional): List of coefficient matrices for combining matrices from `As`.
    Defaults to an empty list.

  Returns:
  array-like: A 3D array representing the stacked local operators.

  Raises:
  ValueError: If both `zs` and `cs` are provided or if neither of them is provided.
  """  
  if checkEmptyList(zs) and checkEmptyList(cs):
      raise ValueError('you must provide either cs or zs')
  elif not checkEmptyList(zs) and not  checkEmptyList(cs):
      raise ValueError('you cannot provide both cs and zs')
      
  if   checkEmptyList(zs):
      T = len(cs[0])
      operators = []
      for t in range(T):
        operators.append(np.sum(np.dstack([cs[i,t]*As[i] for i in range(len(As)) ]) , 2))
      return np.dstack(operators)
  else:      
      T = len(zs)
      operators = []
      for t in range(T):
        operators.append(As[zs[t]])
      return np.dstack(operators)
  
def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't')
    return str_to_change

def c2z(c):
   return np.array([np.argmax(c[:,t]) for t in range(c.shape[1])])
    
def keep_max_vec(vec,k, return_indices = False):
    """
    Keep the k largest values in the vector and set the rest to zero.
    
    Args:
        vec (ndarray): The input vector.
        k (int or float): The number of largest values to keep. If float, it represents a ratio of the vector length.
    
    Returns:
        ndarray: The vector with the k largest values preserved and the rest set to zero.
    """
    vec = vec.flatten()
    if k < 1:
        k = int(k*len(vec))
    lp = np.sort(np.abs(vec))[-k]
    vec[np.abs(vec) < lp] = 0
    if return_indices:
        return vec, np.where(vec != 0)[0]
    return vec


def save_results_switch(folder_save_figs = [] , saving_formats = ['png','svg'], metric_name = 'q_mf', y_model = [], x_model = [], x_step =[], x_look = [], xs = [],
                        c_hat = [], c = [], F_hat = [], F = [], save_npy = False):
  saving_formats = [ '.' +saving_format for saving_format in saving_formats if saving_format[0] != '.']
  if checkEmptyList(folder_save_figs):
      folder_save_figs = os.getcwd()
  if not os.path.exists(folder_save_figs + os.sep + metric_name):
    os.makedirs(folder_save_figs + os.sep + metric_name, exist_ok=False)
  if save_npy:
    np.save(folder_save_figs + os.sep + 'results_%s.npy'%metric_name,locals() )
    

  """
  3d
  """
  fig, axs = plt.subplots(1, 5, figsize = (20,5), subplot_kw={'projection': '3d'})
  plot_3d(y_model.T, ax = axs[0])
  plot_3d(x_model.T, ax = axs[1])
  plot_3d(xs, ax = axs[2])
  plot_3d(x_step, ax = axs[3])
  plot_3d(x_look, ax = axs[4])
  titles = ['$y_{model}$', '$x_{model}$', '$x$', '$\hat{x}^1$', '$\hat{x}^T$']
  [add_labels(ax, zlabel = '\n z \n ',xlabel = '\n x \n ',ylabel = '\n y \n ', title = titles[i]) for i,ax in enumerate(axs)]

  fig.suptitle(metric_name)
  fig.tight_layout()

  for saving_format in saving_formats:
    fig.savefig(folder_save_figs + os.sep + metric_name + os.sep + metric_name + '_3d_recos' + saving_format)

  """
  2d
  """  
  fig, axs = plt.subplots(1, 5, figsize = (20,5), sharey = True, sharex = True)
  axs[0].plot(y_model)
  axs[1].plot(x_model)
  axs[2].plot(xs.T)
  axs[3].plot(x_step.T)
  axs[4].plot(x_look.T)
  titles = ['$y_{model}$', '$x_{model}$', '$x$', '$\hat{x}^1$', '$\hat{x}^T$']
  [add_labels(ax, zlabel = '',xlabel = '\n x \n ',ylabel = '\n y \n ' if i ==0 else '', title = titles[i]) for i,ax in enumerate(axs)]

  fig.suptitle(metric_name)
  fig.tight_layout()

  for saving_format in saving_formats:
    fig.savefig(folder_save_figs + os.sep + metric_name + os.sep + metric_name + '_2d_recos' + saving_format)

  """
  2d nonsharey 
  """  
  fig, axs = plt.subplots(1, 5, figsize = (20,5), sharey = False, sharex = True)
  axs[0].plot(y_model)
  axs[1].plot(x_model)
  axs[2].plot(xs.T)
  axs[3].plot(x_step.T)
  axs[4].plot(x_look.T)
  titles = ['$y_{model}$', '$x_{model}$', '$x$', '$\hat{x}^1$', '$\hat{x}^T$']
  [add_labels(ax, zlabel = '',xlabel = '\n x \n ',ylabel = '\n y \n ' if i ==0 else '', title = titles[i]) for i,ax in enumerate(axs)]

  fig.suptitle(metric_name +' (non shareing y axis)')
  fig.tight_layout()

  for saving_format in saving_formats:
    fig.savefig(folder_save_figs + os.sep + metric_name + os.sep + metric_name + '_2d_recos_non_share_y' + saving_format)

  """
  plot c_hat
  """
  fig, ax  = plt.subplots(2,1,figsize = (40,10), sharey = True, sharex = True)
  sns.heatmap(c_hat, cbar = False, ax = ax[0])
  sns.heatmap(c, cbar = False, ax = ax[1])
  titles = ['$\hat{c}$', '$c$']
  [add_labels(ax_i, xlabel = 'Time', ylabel = '', zlabel = '', title = titles[i]) for  i, ax_i in enumerate(ax)]
  fig.tight_layout()
  for saving_format in saving_formats:
    fig.savefig(folder_save_figs + os.sep + metric_name + os.sep + metric_name + 'c' + saving_format)

  """
  plot f_hat
  """
  fig,axs = plt.subplots(2,len(F), figsize = (len(F)*4, 10))
  [sns.heatmap(F[i], ax = axs[0,i], square = True, annot = True) for i in range(3)]
  [sns.heatmap(F_hat[i], ax = axs[1,i], square = True, annot = True) for i in range(3)]

  
  titles = ['$F_%d$'%(i+1) for i in range(len(F))]
  ylabels = ['$\hat{F}$', '$F$']
  [[add_labels(ax_i, xlabel = '', ylabel = ylabels[row], zlabel = '', title = titles[i]) for  i, ax_i in enumerate(ax)] for row, ax in enumerate(axs)]
  fig.tight_layout()
  for saving_format in saving_formats:
    fig.savefig(folder_save_figs + os.sep + metric_name + os.sep + metric_name + 'F' + saving_format)
    
def apply_hard_thres(mat, axis = 0, k = 0.5):
    """
    Apply hard thresholding to a matrix along a specified axis.

    Args:
        mat (ndarray): The input matrix.
        axis (int, optional): The axis along which to apply the thresholding. Default is 0.
        k (int or float, optional): The number of largest values to keep. If float, it represents a ratio of the matrix length.

    Returns:
        ndarray: The matrix with the hard thresholding applied.

    Raises:
        ValueError: If axis is not 0, 1,2 or -1.
    """
    if np.max(mat.shape) == len(mat.flatten()):
        return keep_max_vec.reshape(mat.shape)
        
    if axis == 0:
        if k > mat.shape[0]:
            k = mat.shape[0] - 1
        if k < 1:
            k =int( k*mat.shape[0])
        return np.hstack(
            [keep_max_vec(mat[:,i], k ).reshape((-1,1)) for i in range(mat.shape[1])]
            )
    elif axis == -1:
        lp = np.sort(np.abs(mat).flatten())[-k]
        mat[np.abs(mat) < lp] = 0
        return mat
    elif axis == 2:
        
        return np.transpose(np.dstack([apply_hard_thres(mat[i,:,:].T, axis = 0, k = k)
                             for i in range(mat.shape[1])]), [2,1,0])#ranspose([2,1,0])
    elif axis == 1:
        return apply_hard_thres(mat.T, axis = 0, k = k).T
    else:
        raise ValueError('axis must be 1, or 0, or -1')

def z2c(z):
  z = z - np.min(z)
  M = np.max(z) +1 #- np.min(z) 
  T = len(z)
  c = np.zeros((M, T))
  for t in range(T):
    c[z[t], t]   =1 
  return c  
    
def reco_ss_slds(As, z, x, type_look = 'step'):
  x = x.T
  M = len(As)
  T = x.shape[1]

  c =  z2c(z)
  if type_look == 'step':
    x_hat = np.hstack([(np.sum(np.dstack([c[i,t]*As[i] for i in range(M)]), 2) @ x[:,t].reshape((-1,1))).reshape((-1,1)) for t in range(T)])
  else:
    x_hat = x[:,0].reshape((-1,1))
    for t in range(T):
      x_hat = np.hstack([ x_hat,( np.sum(np.dstack([c[i,t]*As[i] for i in range(M)]), 2) @ x_hat[:,-1].reshape((-1,1))).reshape((-1,1)) ])
  return x_hat

    
    
    
    
    
def min_diff(nums):
    nums = np.array(nums)       
    diff_mat = nums.reshape((1,-1)) - nums.reshape((-1,1))
    diff_mat[diff_mat == 0] = np.max(np.abs(nums)) + 1
    row, col = np.unravel_index(np.argmin(np.abs(diff_mat)), diff_mat.shape)
    val = nums[row]#,col]
    for i in range(3):
        #mean = np.mean(nums)
        where = np.argmax(np.abs(nums - val))
        nums[where] = val
        print(nums)
    return np.max(nums) - np.min(nums)
    
    
import itertools
def min_diff(nums):
    if len(nums) <= 4:
        return 0
    nums = np.array(nums)       
    missing = itertools.combinations(np.arange(len(nums)), 3)
    lar = [largest_gap(nums,miss) for miss in missing]
    print(lar)
    ret = np.abs(np.nanmin(lar).astype(int))
    print(ret)
    return ret# int(np.nanmin(lar)    )


def largest_gap(nums, miss):
    nums = nums.astype(float)
    nums[np.array(miss)] = np.nan
    return np.nanmax(nums) - np.nanmin(nums)



#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

to_run = False # str2bool(input('to run?'))
# Define the values
theta = np.pi*0.2# was 0.2
T = 100
period_min = 30
period_max = 40
seed = 0
x0 = []
noise_level = 0
making_sure_all = False
offset = []
w_gauss = True # False
wind_gauss  = 10
# Create the dictionary
params = {
    'wind_gauss': wind_gauss,
    'theta': theta,
    'T': T,
    'period_min': period_min,
    'period_max': period_max,
    'seed': seed,
    'x0': x0,
    'noise_level':noise_level,
    'w_gauss':w_gauss,
    'offset':offset,
    'making_sure_all':making_sure_all 
    
}



# Get the full path of the current script
script_path = os.path.abspath(__file__)
# Print the full path
#print("Full path of the script:", script_path)
today = str(datetime2.today()).split()[0].replace('-','_')

if to_run:    
    x, c, c_gauss, F,  x_noisy = create_synth_for_lookahead_slds(**params)
    operators = find_local_operators(F, zs  = [], cs = cc_gaus)
    z_ground_truth =  c2z(cc_gaus)
    np.save('synth_data_for_slds_ground_truth_noise_%d_%s_w_gauss_%s.npy'%(noise_level,today, str(w_gauss)), 
            {'xs':x, 'cs_sharp': cc, 'F':F, 'operators': operators, 'zs':z_ground_truth,
                                                     'params':params, 'script_path':script_path, 'cs':cc_gaus, 'offset':offset})
    
    
    
    

    
    
    
    
    
    
    
    
    