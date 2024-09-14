# -*- coding: utf-8 -*-


# NOTES:
    # !!!!!!!!!! LINEAR !!!!!!!!!!!!:
        #  opt_A, opt_b,b_hats = train_linear_system_opt_A(x_1step, max_ord, Bs_main = [], constraint = [], w_offset = True, cal_offset=True, weights=w, infer_b_way='each', K_b= K_b, w_b = w_b 
#  train_LOOKAHEAD - NON LINEAR
#  

########################################################
# NAIVE FINDER
########################################################
import numpy as np
from scipy.optimize import least_squares
from  basic_function_lookahead  import *
from scipy.linalg import fractional_matrix_power
import seaborn as sns
global today
from datetime import datetime as datetime2
today = str(datetime2.today()).split()[0].replace('-','_')

def find_operators_h1_no_reg(data):
    return np.dstack([ data[:,t+1].reshape((-1,1))  @
                       np.linalg.pinv(data[:,t].reshape((-1,1)) ).reshape((1,-1))  
                      for t in range(data.shape[1]-1 ) ])


def find_operators_h1_l2_reg(data, l2_w, smooth_w):
    if smooth_w == 0:
        return np.dstack([ find_operator_under_l2_or_smoothness(data[:,t+1], data[:,t], l2_w, 
                                                            smooth_w, t = t, A_minus = []) 
                          for t in range(data.shape[1]-1 ) ])
    else:
        As = []
        for t in range(data.shape[1]-1 ):
            if t > 0 :  A_minus  = A_next
            else:       A_minus = []
            
            A_next = find_operator_under_l2_or_smoothness(data[:,t+1], data[:,t], l2_w, 
                                                        smooth_w, t = t, A_minus = A_minus) 
            As.append(A_next)
        return np.dstack(As)

def find_c_from_operators(operators, F):
    F_mat = np.hstack([F_i.flatten().reshape((-1,1)) for F_i in F])
    F_mat_inv = np.linalg.pinv(F_mat)
    operators_stack = np.hstack([operators[:,:,t].flatten().reshape((-1,1)) for t in range(operators.shape[2])])
    return F_mat_inv @ operators_stack


def find_operators_under_l0( data, params_thres = {'thres':1, 'num':True, 'perc':False}):

    if  params_thres['thres'] == 0:
        return find_operators_h1_no_reg(data)
    else:
        As = []
        for t in range(data.shape[1]-1 ):
            A_next = data[:,t+1].reshape((-1,1))  @ np.linalg.pinv(data[:,t].reshape((-1,1)) ).reshape((1,-1)) 
            A_next = keep_thres_only(A_next, direction = 'lower' , **params_thres)                  
              
            As.append(A_next)
        return np.dstack(As)

def k_step_prediction(x, As, K, store_mid = True, t = -1, offset = []):     
    if len(As.shape) == 2 or (len(As.shape) == 3 and As.shape[-1] == 1):
       if (len(As.shape) == 3 and As.shape[-1] == 1):
           As = As[:,:,0]
       return k_step_prediction_linear(x, As, K, store_mid , t , offset)
    else:
       if not checkEmptyList(offset):
           raise ValueError('future offset')
       if is_1d(x):
           x = x.reshape((-1,1))
       if store_mid:
           stores = []  
       for k in range(K):
           print('k')
           print(k)
           
           #stores.append(x)
           x = As[:,:,k] @ x
           if is_1d(x):
               x = x.reshape((-1,1))
    
           if store_mid:
               stores.append(x)    
       if store_mid:
           return x, stores
       return x


def find_operator_under_l1(data, l1_w, seed = 0, params = {} ):
    params ={**{'threshkind':'soft','solver':'spgl1','num_iters':10},**params}
    p = data.shape[0]
    As = []
    for t in range(data.shape[1]-1 ):
        #if per_row:
        A_cur = []
        for el in range(p):
            cur_b = np.array([data[el,t+1]]).reshape((1,-1))
            cur_A = data[:,t].reshape((1,-1))
            A_cur.append(solve_Lasso_style(cur_A, cur_b, l1_w, 
                                           params = params, lasso_params = {}, random_state = seed).reshape((1,-1)))
    

        As.append(np.vstack(A_cur))
        # else:
        #     cur_b = np.array([data[:,t+1]]).reshape((1,-1))
        #     cur_A = data[:,t].reshape((1,-1))
            
            
            
    return np.dstack(As)
        
        
    # raise ValueError('HEREHERE FUTURE')
    # # 
    # if l1_w == 0:
    #     return find_operators_h1_no_reg(data)
    # else:
    #     As = []
    #     for t in range(data.shape[1]-1 ):
    #         if t > 0 :  A_minus  = A_next
    #         else:       A_minus = []
            
    #         A_next = find_operator_under_l2_or_smoothness(data[:,t+1], data[:,t], l2_w, 
    #                                                     smooth_w, t = t, A_minus = A_minus) 
    #         As.append(A_next)
    #     return np.dstack(As)

    
    
    
    
def find_operator_under_l2_or_smoothness( y_plus, y_minus, l2_w = 0, smooth_w = 0, t = 0,
                                         A_minus = [], reeval = False, mask = []):
    if reeval and checkEmptyList(mask):
        raise ValueError('you must provide a mask if reeval')
    # this function is not being durectly called
    if reeval:
        A_minus[mask == False] = 0
    if is_1d(y_plus):
        y_plus = y_plus.reshape((-1,1))
        
    if is_1d(y_minus):
        y_minus = y_minus.reshape((-1,1))

    if (l2_w == 0 and smooth_w == 0) or (t == 0 and l2_w == 0):
        print('pay attention - no reg')
        return y_plus  @ np.linalg.pinv( y_minus ).reshape((1,-1))  
                           
    if smooth_w > 0 and checkEmptyList(A_minus) and t > 0:
        raise ValueError('you must provide A_minus')
    else:
        if is_1d(y_plus):
            A_shape = (len(y_plus.flatten()), len(y_plus.flatten()))
        else: 
            A_shape = (y_plus.shape[0], y_plus.shape[0])
        
        if l2_w > 0 and smooth_w > 0 and t > 0:
            plus_addition = np.hstack([np.zeros(A_shape)*l2_w , A_minus*smooth_w ])
            minus_addition = np.hstack( [np.eye(A_shape[0]) , np.eye(A_shape[0])*smooth_w ])
            
        elif  smooth_w > 0 and t > 0:        
            plus_addition = A_minus*smooth_w
            minus_addition = np.eye(A_shape[0])*smooth_w 
        
        elif  l2_w > 0:        
            plus_addition = np.zeros(A_shape)*l2_w
            minus_addition = np.eye(A_shape[0])*l2_w
            
            
        #print(plus_addition.shape )
        #print(y_plus.shape)
        
        y_plus = np.hstack([y_plus, plus_addition ] )    
        
        y_minus = np.hstack(    [y_minus, minus_addition ]   )
        
        #print(y_plus)
        #print(y_minus)
    #A1 = np.linalg.lstsq(y_minus.T, y_plus.T)[0]   
    #A2 =
    #print(A2)
    #print('-----------------------')
    return    y_plus @ np.linalg.pinv(y_minus)  
    








def plot_elements_of_mov( mat,  ax = [] ,  fig = [],colors = [], legend_params = {}, plot_params = {}, to_legend = False, 
                         type_plot = 'plot'):
    if checkEmptyList(colors):
        colors = create_colors(len(mat[:,:,0].flatten()))
    if is_1d(colors):
        colors = colors.reshape((mat[:,:,0].shape[0],mat[:,:,0].shape[1]))
    else:
        colors = colors.reshape((mat[:,:,0].shape[0],mat[:,:,0].shape[1],3))
    # 3d mat
    if checkEmptyList(ax):
        fig, ax = plt.subplots()
    if  type_plot == 'plot':
        [
         
         [
          ax.plot( mat[i,j,:], color = colors[i,j], label = '%d'%(mat.shape[0]*j + i), **plot_params)
          
          for j in range(mat.shape[1])
          ] for i in range(mat.shape[0])
         
         ] 
        
        if to_legend:
            ax.legend(**legend_params)
    elif  type_plot == 'heatmap':
        mat_2d = np.vstack([
         
         np.vstack([
           mat[i,j,:]
          
          for j in range(mat.shape[1])
          ]) for i in range(mat.shape[0])
         
         ] )
        sns.heatmap(mat_2d, ax = ax, **plot_params)
   


def find_weight(k, e, sigma1 = 12, sigma2 = 1.1, norm_w = True, norm_direction = 1):    
    """
    Calculate the weight using a specified formula.
    
    Parameters:
    - k (float): Input parameter.
    - e (float): Input parameter.
    - sigma1 (float, optional): Standard deviation parameter for k. Default is 12.
    - sigma2 (float, optional): Exponent parameter for (1 + |e|). Default is 1.1.
    
    Returns:
    float: Weight calculated based on the input parameters.
    """
    w = np.exp(-k*sigma1)*(1+np.abs(e))**sigma2
    w[np.isnan(w)] = 0
    if (w == 0).all():
        raise ValueError('all weights are 0!')
    if norm_w:
        if norm_direction == -1:
            w /= w.sum()
        elif norm_direction == 0:
            w /= w.sum(0).reshape((1,-1))
        elif norm_direction == 1:
            w /= w.sum(1).reshape((-1,1))
        else:
            raise ValueError('Invalid direction to norm!!')
            
    return w

def reevaluate_A_under_mask(y_plus, y_minus, A_mask, constraint_i = '', w_reg_i = ''):
    """
    Reevaluate matrix A based on masks and input vectors.
    
    Parameters:
    - y_plus (numpy.ndarray): Vector with positive values.
    - y_minus (numpy.ndarray): Vector with negative values.
    - A_mask (numpy.ndarray): Mask specifying the structure of matrix A.
    
    Returns:
    numpy.ndarray: Reevaluated matrix A based on the provided masks and vectors.
    """
    if 0 in A_mask:
        A_mask = A_mask != 0
    """
    if the maske covers all
    """
    if np.sum(A_mask) == len(A_mask.flatten()): 
        if is_1d(y_plus) and is_1d(y_minus):
            return y_plus.reshape((-1,1))  @ np.linalg.pinv(y_minus).reshape((1,-1))
        return y_plus  @ np.linalg.pinv(y_minus)
    
    else:
        
        A_new = np.zeros(A_mask.shape)
        for row in range(A_mask.shape[0]) :
            cols = np.where(A_mask[row])[0]
            y_minus_row = y_minus[A_mask[row]]
            if is_1d(y_plus) and is_1d(y_minus):
                gram = np.outer(y_minus_row, y_minus_row)     
                A_row  = y_plus[row]*y_minus_row.reshape((1,-1)) @ np.linalg.pinv(gram) 
            else:
                A_row  = y_plus[row]  @ np.linalg.pinv(y_minus_row ) 
                
            A_new[row, cols] = A_row

        return A_new



def infer_A_under_constraint_under_period(data, K, As,
                               constraint = [], 
                               w_reg = {}, params = {},
                               reeval = True, is1d_dir = 0,
                               given_periods = [], sigma1 = 0.8, sigma2 = 4, norm_w = True,
                              A_former = [], e_thres_for_mid_periods = 0.01, A_between_period_method = 'smooth'):
    #A_between_period_method can be 'smooth' or 'decomposition'
    """
    Infer the matrix A under specified constraints and within given periods.
    
    Parameters:
    - data (numpy.ndarray): Input data matrix.
    - K (int): Lookahead parameter.
    - As (numpy.ndarray): Matrix of As for prediction.
    - constraint (list): List of constraints to apply during training.
    - w_reg (dict): Dictionary of regularization weights.
    - params (dict): Additional parameters for constraint inference.
    - reeval (bool): Reevaluate constraints during training if True.
    - is1d_dir (int): 1D direction parameter.
    - given_periods (list): List of periods for linear dynamics.
    - sigma1 (float): Sigma parameter for weight calculation.
    - sigma2 (float): Sigma parameter for weight calculation.
    - norm_w (bool): Normalize weights if True.
    - A_former (numpy.ndarray): Former matrix A.
    - e_thres_for_mid_periods (float): Error threshold for mid periods.
    - A_between_period_method (str): Method for handling A between periods ('smooth' or 'decomposition').
    
    Returns:
    - As (numpy.ndarray): Updated matrix As.
    """
    if given_periods[0][0] != 0:
        raise ValueError('given periods must start at 0')
    if checkEmptyList(given_periods):
        raise ValueError('given_periods is empty')
    if 'smooth' not in constraint:
        print('smooth not in constraint although you asked for periods')
        input('ok?!')    
    else:
        data_in_periods = []
        As_in_period = []
        for period_num, period in enumerate(given_periods):
            
            cur_data = data[:,period[0]:period[1]]
            data_in_periods.append(cur_data)
            cur_As = data[:,period[0]:period[1] - 1]
            x, stores = k_step_prediction(cur_data, cur_As, K, store_mid = True, t = -1)
            stores_3d = np.dstack(stores)
            """
            error under each order
            """
            es = np.vstack([np.sum((x_i - cur_data)**2, 0) for  x_i in stores]) # for eachtime point
            
        
            w = find_weight(np.arange(1,K+1), es.sum(1), sigma1 = sigma1, sigma2 = sigma2, norm_w = norm_w)  # different orders
            
            
                
            # RE-WEIGH EACH ORDER EST.
            stored_3d_weighted = stores_3d * w.reshape((1,-1,1))  
                
            # FIND PROP FOR EVERY ORDER            
            A_hat = find_A_based_on_multiple_orders_for_period(stored_3d_weighted, cur_data,
                                                               constraint =  constraint, w_reg = w_reg, params = params,
                                          reeval = reeval , is1d_dir = is1d_dir, A_former = A_former, t = period_num)
            
            # this is a vector of dim X num orders X 2. data plus is just the data. data minues is the estimation
            #data_plus_data_minus_w = data_plus_data_minus  * w_t.reshape((1,-1,1))  
            
            #constraint , w_reg,, params = {},
            #                            reeval = True , is1d_dir = 0, A_former = [], t = 0
            As_in_period.append(A_hat)
            As[:,:, period[0]: period[1]] = A_hat.reshape((-1, A_hat.shape[1], 1))
      
            
        """
        infer mid points
        """
        
        xs, _ = [k_step_prediction(data_i, As_in_period[i], K, store_mid = False, t = -1) for i, data_i in enumerate(data_in_periods)]
        e = np.mean([((xs[i] - data_i)**2).mean()  for i, data_i in enumerate(data_in_periods)])
        
        F = As_in_period.copy()
        cs = np.zeros((len(F), As.shape[2]))
        for period_num, period in periods:
            cs[:, period[0]: period[1]] = create_sparse_lambda(len(F), period_num)
            
        if e < e_thres_for_mid_periods:
            between_periods = [[period[1], given_periods[i+1][0]] for i, period in enumerate(given_periods[:-1])] 
            if A_between_period_method == 'smooth':
                
                
                for between_period in between_periods:
                    #period_0 = 
                    #period_1 = 
                    len_period = between_period[1] -  between_period[0]
                    A_former = As[:,:, between_period[0] - 1]
                    data_between = data[:,between_period[0]-1:between_period[1]+1]
                    #data_between = data[:,between_period[0]:between_period[1]]
                        
                    for t_relative, t_exact in enumerate(between_period[0]-1, between_period[1]+1):                     
                            K_i = np.min([K, t_relative])
                            #t_start = np.max([t_relative - 1,0])         
                            #t_end = np.min([t_relative + K - 1, len_period  ])
                            
                            # PREDICT ONE STEP FROM X_T
                            #data_next = one_step_prediction(data_between[:,t_relative], A_former, t = -1, k = -1, t_start = -1, t_end = -1)                         
                            A_former = infer_A_under_constraint(data_between[:,t_relative+1],data_between[:,t_relative],
                                                           constraint = constraint, 
                                                           w_reg = w_reg, params = params,
                                                           reeval = reeval, is1d_dir = is1d_dir, A_former = A_former, t = t-1, 
                                                           given_periods = given_periods)
                            
                            As[:,:,t_exact] = A_former   
                    cs = find_c_from_operators(As[:,:,between_period[0]-1:between_period[1]+1], F)
                            
                            
        elif 'deco' in A_between_period_method:
            for between_period in between_periods:
                data_between = data[:,between_period[0]-1:between_period[1]+1]
                As_between, cs_between = find_A_decomposed_K_steps(data_between, F, K, sigma1, sigma2, A_former = A_former,  
                                                                   w_reg = w_reg, params = params, 
                                              reeval = reeval, reweigh_As_func = np.nanmedian, norm_w = True )
                cs[:,between_period[0]-1:between_period[1]] = cs_between
                As[:,:,between_period[0]-1:between_period[1]] = As_between
            
            
        else:
            raise ValueError('A_between_period_method is %s but must be "smooth" or "deco"'%A_between_period_method)
                                 
        return As, cs
    
    
    
def find_A_decomposed_1_step_1_time(data_plus, data_minus, F, A_former = [],  w_reg = {},
                                    params = {}, reeval = True , smooth_w = 0, cur_reco_t = [], lasso_params = {}, l1_w = 0):
    if not checkEmptyList( A_former) and 'smooth' in constraint:
        raise ValueError('future extension! smoothness in decomposition')
        
    if not is_1d(data_minus):
        raise ValueError('data_minus is not 1d')
        
    elif not is_1d(data_plus):
        raise ValueError('data_plus is not 1d')
        
    else:
        #find_A_given_basis_operators(x, basis_Fs, A_former, = [],  
        #                             w_reg = w_reg, params = params, reeval = reeval )
        Fx = np.hstack([(F_i @ data_minus.reshape((-1,1))).reshape((-1,1)) for F_i in F])
        # find Cs
        #print(Fx.shape)
        """
        FIND C
        """
        left_side = Fx
        right_side = data_plus.reshape((-1,1))
        if smooth_w > 0 and not checkEmptyList(cur_reco_t):
            left_side = np.vstack([left_side, np.eye(left_side.shape[1])])
            right_side = np.vstack([right_side, cur_reco_t.reshape((-1,1))])
        if l1_w == 0:
            cs_t = np.linalg.pinv(left_side) @ right_side       
        else:
            cs_t = solve_Lasso_style(left_side, right_side, l1_w, 
                                           params = params, lasso_params = lasso_params, random_state = 0).reshape((1,-1))
    

            
        cs_t = cs_t.reshape((-1,1))
        A_t = np.sum(np.dstack([F_i*cs_t[i] for i,F_i in enumerate(F)]), 2)
        cur_reco_t = A_t @ data_minus.reshape((-1,1))
        return A_t, cs_t, cur_reco_t
        

def find_A_decomposed_1_step_period(data_plus, data_minus, F, A_former = [],  w_reg = {}, params = {}, 
                                    reeval = True, smooth_w = 0, lasso_params = {} , l1_w = 0, c_start = []):
    if not checkEmptyList( A_former) and 'smooth' in constraint:
        raise ValueError('future extension! smoothness in decomposition')
    if data_plus.shape[1] != data_minus.shape[1]:
        raise ValueError('data plus must have the same dim as data_minus, but data_plus.shape = %s, data_minus.shape = %s'%(str(data_plus.shape), str(data_minus.shape)))
        
        
    if  is_1d(data_minus):
        return  find_A_decomposed_1_step_1_time(data_plus, data_minus, F,A_former = A_former,  
                                                w_reg = w_reg, params = params, reeval = reeval , l1_w = l1_w)
    #    raise ValueError('data_minus is not 1d')

        
    else:
        # THIS IS A LIST OF RECOS. Each is 2d mat. 
        #print(data_minus.shape)
        #print('!')
        # Fx = [ [F_i @ data_minus[:,t].reshape((-1,1)) for F_i in F] for t in range(data_minus.shape[1])]
        
        # # 
        # cs = np.hstack([(np.linalg.pinv(Fx[t]) @ data_plus[:,t].reshape((-1,1))).reshape((-1,1)) for t in range(data_minus.shape[1])])
   
        # A = np.dstack([np.sum(np.dstack([F_i*cs[i,t] for i,F_i in enumerate(F)]), 2) for t in range(data_minus.shape[1])])
        # cur_reco = np.hstack([(A[:,:,t] @ data_minus[:,t].reshape((-1,1))).reshape((-1,1)) for t in range(A.shape[2])])
        A =        []
        cs =      []
        cur_reco = []
        cs_t =  c_start #[]
        for t in range(data_minus.shape[1]):
            A_t, cs_t, cur_reco_t  = find_A_decomposed_1_step_1_time(data_plus[:,t], data_minus[:,t], F, A_former,  w_reg,
                                                                     params, reeval, smooth_w = smooth_w, cur_reco_t = cs_t,
                                                                     lasso_params = lasso_params  , l1_w = l1_w)
            A.append(A_t)
            # cs_t here isa vector of len M
            cs.append(cs_t)
            cur_reco.append(cur_reco_t)
            
        A = np.dstack(A)
        # cs is a matrix of M X T
        cs = np.hstack(cs)
        cur_reco = np.hstack(cur_reco)
        return A, cs, cur_reco
        
    
def try_optimize(c_t, F, x_noisy_t, x_next, c_t_know, c_former, reco_so_far, t ):
    Fx = np.hstack([(F[i]@x_noisy_t.reshape((-1,1))).reshape((-1,1)) for i in range(len(F))])
    if len(reco_so_far) > 0:
        F_so_far = np.hstack([(F[i]@reco_so_far.reshape((-1,1))).reshape((-1,1)) for i in range(len(F))])
        addi2 =  0.01*(t**1.2)*(x_next - F_so_far @ c_t)**2
    else:
        addi2= 0
    if not checkEmptyList(c_former):
        addi = 0.5*(c_t - c_former)**2
    else:
        addi = 0
    objective =  1.3*np.abs(c_t) + 1.5*np.abs(c_t - c_t_know)  + addi+addi2 + (x_next - Fx @ c_t)**2 #c_t**2 + 
    return objective.sum()

    
    
    
    
    
def d3tod323(mat)   :
    mat_2d = np.vstack([
     
     np.vstack([
       mat[i,j,:]
      
      for j in range(mat.shape[1])
      ]) for i in range(mat.shape[0])
     
     ] )
    print(mat_2d.shape)
    return mat_2d
        
    
    
    
    
    
    
    
    
    
def find_A_fractional_deptacated(k, B):
    """
    Find the matrix A such that A^k = B.

    Parameters:
    - k (float): Power value.
    - B (array-like): Target matrix.

    Returns:
    array-like: Matrix A satisfying A^k = B.
    """
    A = fractional_matrix_power(B, 1/k)
    return A

def infer_A_under_constraint(y_plus, y_minus, constraint = ['l0'], w_reg = 3, params = {},
                              reeval = True , is1d_dir = 0, A_former = [], t = 0):
    #future not l0   
    
    #if type(constraint) != type(w_reg) and  (isinstance(constraint, list) and len(constraint) != 1) :
    #    print('w_reg and cnostraing need to be of the same type. but %s, %s'%(str(constraint), str(w_reg)))
        
    
    if checkEmptyList(A_former) and 'smooth' in constraint and t > 0:
        raise ValueError('?!?!?!')
        
    if is_1d(y_plus):
        if is1d_dir == 0:
            y_plus = y_plus.reshape((-1,1 ))
        else:
            y_plus = y_plus.reshape((1,-1 ))
           
            
    if is_1d(y_minus):
        if is1d_dir == 0:
            y_minus = y_minus.reshape((-1,1 ))
            shape_inv = (1,-1)
        else:
            y_minus = y_minus.reshape((1,-1 ))
            shape_inv = (1,-1)
    else:
        shape_inv = y_minus.shape[::-1]
       
    try:    
        A_hat = y_plus @ np.linalg.pinv(y_minus)
    except:
        print('y plus ?!')
        print(y_plus)
        A_hat = y_plus @ np.linalg.pinv(y_minus + np.random.rand(*y_minus.shape)*0.01)
        
    for  constraint_i in  constraint:
        w_reg_i = w_reg[constraint_i]
        A_hat = apply_constraint_after_order(y_plus, y_minus, constraint_i, w_reg_i, A_former = A_former, t = t, A_hat = A_hat, reeval = True)    
    return A_hat


def find_Bs_for_dynamics(data, K, constraint = [], w_reg = [], params = {},
                              reeval = True , is1d_dir = 0, A_former = [], t = 0, 
                              w_offset = True, addi = ''):
    # Bs  is a list of elements of B. Each B (element of Bs) is the operator y_t  = B y_{t-k}. 
    Bs = []
    for k_i in range(1, K+ 1):
        y_plus = data[:, k_i:]  
        y_minus = data[:, :-k_i]
        if w_offset:
            y_minus_expanded = np.vstack([y_minus, np.ones((1, y_minus.shape[1] )) ])
        else:
            y_minus_expanded = y_minus
        
        # Bs  is a list of elements of B. Each B (element of Bs) is the operator y_t  = B y_{t-k}. 
        B =  infer_A_under_constraint(y_plus, y_minus_expanded, constraint, w_reg, params,
                                      reeval , is1d_dir, A_former = A_former, t = 0)
        
        Bs.append(B)
    if w_offset:
        Bs_main = [B[:,:-1] for B in Bs]
    else:
        Bs_main = Bs.copy()
    return Bs, Bs_main
    
from scipy.optimize import minimize

def objective_function(A, Bs_main, weights = [], with_identity = False):
    A = A.reshape(Bs_main[0].shape)
    if checkEmptyList(weights):
        weights = np.ones(len(Bs_main))
    if not with_identity:
        terms = [weights[k]*(np.linalg.matrix_power(A, k + 1) - B_i).T @ (np.linalg.matrix_power(A, k + 1) - B_i)
                 for k, B_i  in enumerate(Bs_main)]
    else:
        dim = A.shape[0]
        terms = [weights[k]*(np.linalg.matrix_power(A, k + 1) - B_i - np.eye(dim)).T @ (np.linalg.matrix_power(A, k + 1) - B_i - np.eye(dim)) for k, B_i  in enumerate(Bs_main)]
    #print(terms)
    objective =  np.sum(np.abs(np.dstack(terms)))
    return objective
    #term1 = A - A1
    #term2 = A @ A - A2
    #return np.sum(term1 + term2)


def optimize_A_using_optimizer(Bs_main, A0 = [], weights = [], with_identity = False):
    #
    if checkEmptyList(A0):
        # Initial guess for A
        A0 = np.zeros_like(Bs_main[0])
    
    # Minimize the objective function
    #print(A0.shape)
    result = minimize(objective_function, A0, method='BFGS', args = (Bs_main, weights,  with_identity))
    
    # The optimized matrix A
    optimized_A = result.x.reshape(Bs_main[0].shape)
    
    #print("Optimized A:")
    return optimized_A
    
    
#use scipy minimize!!
def poly_cost(lambda_i, lambda_powers, weights):
    return np.sum([w*(lambda_i**(i+1) - lambda_powers[i])**2 for i, w in enumerate(weights)])

def lambda_powers2coeffs(lambda_powers = [], weights = []):
    # lambda powers is a list of lambda1, lambda1**2, lambda1**3
    # solves (lambda - lambda1)**2 + (lambda**2 - lambda2)**2 + ....
    if checkEmptyList(weights):
        weights = np.ones(len(lambda_powers))
    if isinstance(lambda_powers,list ):
        lambda_powers = np.array(lambda_powers)
        
    # last coeffs 
    #free_el = np.sum((weights*lambda_powers)**2)
    
    # power 2,4, 6, 8...
    #lambda_coeffs = lists2list([[w] + [0] for w in weights])[::-1] + [0,0]
    
    
    # powers 1,2,3,4....
    #addi = [-2*weights[w_i]*lambda_p for w_i, lambda_p in enumerate(lambda_powers)][::-1] + [0]
    
    #full_coeffs = np.array(lambda_coeffs).astype(complex) 
    #full_coeffs[-len(lambda_powers)-1:] += np.array(addi).astype(complex)
    #full_coeffs[-1] = free_el
    #roots = np.roots(full_coeffs)
    initial_guess = lambda_powers[0]
    roots = minimize(poly_cost, args = (lambda_powers, weights), x0 = initial_guess).x
    
    # take sol closer to lmabda 0
    if len(roots) > 1:
        sol = roots[np.argmin(np.abs(roots - lambda_powers[0]))]
    else:
        sol = roots[0]
    return sol,  roots # free_el, lambda_coeffs, addi, full_coeffs,

    
def lambda_powers2coeffs_depracated(lambda_powers = [], weights = []):
    # lambda powers is a list of lambda1, lambda1**2, lambda1**3
    # solves (lambda - lambda1)**2 + (lambda**2 - lambda2)**2 + ....
    if checkEmptyList(weights):
        weights = np.ones(len(lambda_powers))
    if isinstance(lambda_powers,list ):
        lambda_powers = np.array(lambda_powers)
        
    # last coeffs 
    free_el = np.sum((weights*lambda_powers)**2)
    
    # power 2,4, 6, 8...
    lambda_coeffs = lists2list([[w] + [0] for w in weights])[::-1] + [0,0]
    
    
    # powers 1,2,3,4....
    addi = [-2*weights[w_i]*lambda_p for w_i, lambda_p in enumerate(lambda_powers)][::-1] + [0]
    
    full_coeffs = np.array(lambda_coeffs).astype(complex) 
    full_coeffs[-len(lambda_powers)-1:] += np.array(addi).astype(complex)
    full_coeffs[-1] = free_el
    roots = np.roots(full_coeffs)
    print(roots)
    print('val eq root')

    k = len(full_coeffs)
    #print([weights[k-i]*(root**(k-i-)-lambda_powers[k-i])**2  for i in range(len(weights))])
    for j in range(len(roots)):
        print(np.sum([full_coeffs[i]*roots[j]**(k-i-1) for i in range(k)           ]))
    print('???????????????')
    print(roots)
    print(len(roots))
    print(len(np.unique(roots)))
    print('coeffs')
    print(full_coeffs)
    input('?!')
    # take sol closer to lmabda 0
    if len(roots) > 1:
        sol = roots[np.argmin(np.abs(roots - lambda_powers[0]))]
    else:
        sol = roots[0]
    return sol, free_el, lambda_coeffs, addi, full_coeffs, roots
    
def find_best_lambdas(lambdas, weights = [], matched_evecs = [], roots_full = []):
    # lambdas is a list of numpy arrays. Each array is all evals of that order (each array len p, ovearll K arrays). Fist list is order 1.
    # vector with len K
    
    if checkEmptyList(weights):
        weights = np.ones(len(lambdas))
    
    #weights *= np.exp(-0.1*np.arange(len(lambdas))[::-1])
    weights = weights / np.max(weights)
    K = len(weights)
    p = len(lambdas[0])
    lambdas_best = []
    roots_full = []
    #print('start reco')
    for p_i in range(p):
        lambda_powers = np.array([lambdas_i[p_i] for lambdas_i in lambdas])        
        #lambda_best, free_el, lambda_coeffs, addi, full_coeffs, roots = lambda_powers2coeffs(lambda_powers, weights)
        lambda_best, roots = lambda_powers2coeffs(lambda_powers, weights)
        lambdas_best.append(lambda_best)
        roots_full.append(roots)
    # eigen deco
    if isinstance(matched_evecs, list):
        matched_evecs = matched_evecs[0]
    deco  = eigen_dec(np.array(lambdas_best),  matched_evecs)
    # private decos
    each_deco =  [eigen_dec(np.array(lambdas_i),  matched_evecs) for lambdas_i in lambdas]
    
    """
    deco for each solution
    """
    print('find decos')
    combinations_r = list(itertools.product(*roots_full))
    decos_all_roots = [eigen_dec(np.array(combination),  matched_evecs) for combination in combinations_r]
    
    return deco, lambdas_best,  each_deco, decos_all_roots, roots_full
    
    
def reorg_evecs_all_orders(Bs, to_plot = True, to_save = True, save_path = '.', 
                           save_formats = ['.png','.svg'], fig = [], axs = [], with_signs = False, to_abs = True):    
    addi = today + '_%s'%str(to_abs)
    # Bs is a list of matrices A, A^2, A^3....
    evecs = []
    evals = []
    """
    FIND EVECS AND EVALS
    """
    for B in Bs:
        w, v = np.linalg.eig(B)    
        evecs.append(v)
        evals.append(w)
        
    """
    REORDER EVECS AND EVALS BY FIRST COLUMN
    """
    matches_and_signs = [find_matching_columns(evecs[0], evec, to_abs = to_abs,i= i) for i,evec in enumerate(evecs)]
    matches = [np.vstack(el[0]) for el in matches_and_signs]
    if with_signs:
        signs = [el[1] for el in matches_and_signs]
    else:
        signs = [np.ones((1,Bs[0].shape[1])) for _ in matches_and_signs]
    #matches = [np.vstack(find_matching_columns(evecs[0], evec)) for evec in evecs]
    matched_evecs = []
    matched_evals = []
    dim = Bs[0].shape[0]
    
    for count, evec in enumerate(evecs):
        cur_match = matches[count]
        eval_c = evals[count]
        #cur_dict = {cur_match[0]:cur_match[1] for cur_match in matches}
        # print('--------===============')
        # print(evec.shape)
        # print(cur_match)
        
        """
        order_matches = indices of 2nd mat
        """
        order_matches = np.array([cur_match[cur_match[:,0] == j, 1] for j in range(dim)]).flatten()
        
        # print( order_matches)
        # print('----------------------------')
        signs[count] = signs[count][order_matches]
        matched_eval = eval_c[order_matches]*signs[count]
        #print(signs[count])
        matched_eval = np.array([eval_c_i.conj() if signs[count][i] < 0 else eval_c_i for i,eval_c_i in enumerate(matched_eval) ])
        
        matched_evec = evec[:, order_matches]*signs[count].reshape((1,-1))
        for i in range(matched_evec.shape[1]):
            #if count == 8:
            #print(signs[count][i])
            #print(evec[:, order_matches])
            if signs[count][i] < 0:
                matched_evec[:,i] = matched_evec[:,i].conj()
            
            
        
        matched_evecs.append(matched_evec)
        matched_evals.append(matched_eval)
        
    
    
    
    if to_plot:
        match_cols, disparities, match_cols_dict = eval_matches(matched_evecs, to_plot = True)
        print('plot!!!!!!!!!!!!!!!!!!')
        B_min = np.min([B.min() for B in Bs])
        B_max = np.max([B.max() for B in Bs])
        
        v_min = np.min([np.real(B.min()) for B in evecs])
        v_max = np.max([np.real(B.max()) for B in evecs])
        
        v_min_r = np.min([np.imag(B).min() for B in evecs])
        v_max_r = np.max([np.imag(B).max() for B in evecs])        
        

        if checkEmptyList(axs):            
            fig, axs = plt.subplots(5, len(Bs), figsize = (len(Bs)*5, 15),sharey = True)
            
            
        [sns.heatmap(B, ax = axs[0,i], vmin = B_min, vmax = B_max) for i,B in enumerate(Bs)]        
        [add_labels(ax, xlabel = 'x-y-z', ylabel = 'x-y-z', title = '$A^%d$'%(i+1), title_params = {'fontsize': 12}) for i, ax in enumerate(axs[0])]
        
        [sns.heatmap(np.real(B), ax = axs[1,i], vmin = v_min, vmax = v_max) for i,B in enumerate(evecs)]        
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals', title = 'Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs[1])]

        [sns.heatmap(np.real(B), ax = axs[3,i], vmin = v_min, vmax = v_max) for i,B in enumerate(matched_evecs)]        
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals (match)', title = '(matched) Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs[3])]  
        
        [sns.heatmap(np.imag(B), ax = axs[2,i], vmin = v_min_r, vmax = v_max_r) for i,B in enumerate(evecs)]        
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals', title = 'IM Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs[2])]

        [sns.heatmap(np.imag(B), ax = axs[4,i], vmin = v_min_r, vmax = v_max_r) for i,B in enumerate(matched_evecs)]        
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals (match)', title = 'IM (matched) Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs[4])]  
        
        
        # """
        # plot evals
        # """
        
                
        v_min = np.min([np.real(B.min()) for B in evecs])
        v_max = np.max([np.real(B.max()) for B in evecs])
        
        v_min_r = np.min([np.imag(B.min()) for B in evecs])
        v_max_r = np.max([np.imag(B.max()) for B in evecs])        
        
        
        np.random.seed(0)
        colors = np.random.rand(len(evals), 3)
        colors = [tuple(colors[i]) for i in range(colors.shape[0])]
        fig_eval, axs_eval = plt.subplots(2, len(Bs), figsize = (len(Bs)*5, 15),sharey = True, sharex = True)
                    

        [axs_eval[0,i].scatter(np.real(B), np.imag(B), color = colors) for i,B in enumerate(evals)]        
        #[axs[0,i].plot([0,np.real(B)], [0,np.imag(B)]) for i,B in enumerate(evals)] 
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals', title = 'Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs_eval[0])]


        #print(matched_evals[0])
        [axs_eval[1,i].scatter(np.real(B), np.imag(B), color = colors) for i,B in enumerate(matched_evals)]   
        #[axs[1,i].plot([0,np.real(B)], [0,np.imag(B)]) for i,B in enumerate(matched_evals)] 
        [add_labels(ax, xlabel = 'evec #', ylabel = 'evecs vals (match)', title = 'IM (matched) Evecs Order k = %d'%i, title_params = {'fontsize': 12}) for i, ax in enumerate(axs_eval[1])]  
        
        
        
        
        fig.tight_layout()
        if to_save:
            fig.suptitle(save_path + os.sep + 'evecs_%s'%addi + save_formats[0])
            [fig.savefig(save_path + os.sep + 'evecs_%s'%addi + save_format) for save_format in save_formats]
            
            fig_evals.suptitle(save_path + os.sep + 'evals_%s'%addi + save_formats[0])
            [fig_evals.savefig(save_path + os.sep + 'evals_%s'%addi + save_format) for save_format in save_formats]
    else:
        match_cols, disparities = [],[]
    return matched_evecs, matched_evals, match_cols, disparities, signs
    
    
    
            
def eval_matches(matched_Ps, to_plot = True, save_path = '.', save_formats = ['.png','.svg']):   
    addi = today
    combs = list(itertools.combinations(range(len(matched_Ps)), 2))
    #print(list(combs))
    #[find_matching_columns(matched_Ps[comb[0]],  matched_Ps[comb[1]] )[0] for comb in combs]
    match_cols = [np.vstack(find_matching_columns(matched_Ps[comb[0]],  matched_Ps[comb[1]] )[0]) for comb in combs.copy()]
    
    #print(list(combs))
    match_cols_dict = {tuple(comb):np.vstack(find_matching_columns(matched_Ps[comb[0]],  matched_Ps[comb[1]] )[0]) for comb in list(combs)}
    #print(match_cols_dict)
    disparities = [(match_col[:,0] != match_col[:,1]).sum() for match_col in match_cols]
    print('Columns do not match %d times'%np.mean(disparities))
    
    if to_plot:
        each = int(np.sqrt(len(match_cols))) + 1
        fig, axs = plt.subplots(each-2,each+2, figsize = (each*2,each*7), sharey = True)
        axs = axs.flatten()
        [sns.heatmap(match_cols_i, vmin = 0, vmax = matched_Ps[0].shape[0], ax = axs[i], annot = True, square = True, cbar = False)      for i, match_cols_i in enumerate( match_cols )]
        fig.suptitle('Columns Matching')
        fig.tight_layout()
        [add_labels(ax, xlabel = 'Pair', ylabel = '') for ax in axs]
        
        fig.tight_layout()
        if to_save:
            fig.suptitle(save_path + os.sep + 'pairs_%s'%addi + save_formats[0])
            [fig.savefig(save_path + os.sep + 'pairs_%s'%addi + save_format) for save_format in save_formats]

        
        
        fig, axs = plt.subplots()
        sns.histplot(np.abs(disparities), ax = axs, bins = matched_Ps[0].shape[0] ,discrete = True)
        add_labels(axs, xlabel = 'Disparities', ylabel = 'Count')
        
        fig.tight_layout()
        if to_save:
            fig.suptitle(save_path + os.sep + 'Disparities_%s'%addi + save_formats[0])
            [fig.savefig(save_path + os.sep + 'Disparities_%s'%addi + save_format) for save_format in save_formats]

        
        fig, axs = plt.subplots()        
        [axs.scatter(match_cols_i[:,0], match_cols_i[:,1])  for i, match_cols_i in enumerate(match_cols)]
        add_labels(axs, xlabel = 'pair 1', ylabel = 'Pair 2')
        fig.suptitle('Columns Matching')
        fig.tight_layout()
        
        fig.tight_layout()
        if to_save:
            
            fig.suptitle(save_path + os.sep + 'match_%s'%addi + save_formats[0])
            [fig.savefig(save_path + os.sep + 'match_%s'%addi + save_format) for save_format in save_formats]
        
    return match_cols, disparities, match_cols_dict
        

        
        
        


def find_matching_columns(P, P2, to_abs = True, i = 0):    
    # Compute the negative correlation matrix as a cost matrix
    #cost_matrix = -np.corrcoef(P, P2)[:len(P.T), len(P.T):]
    # print('-----------')
    # print(P.shape)
    # print(P2.shape)
    cost_matrix = -cosine_sim_complex_vectors(P, P2) + 1
    #print(cost_matrix)
    #print('!!!!!!!!!!')
    # Solve the linear assignment problem
    if to_abs:
        row_indices, col_indices = linear_sum_assignment(np.abs(cost_matrix))
    else:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
        
    signs = np.array([cost_matrix[row, col] <= 1 for row, col in zip(row_indices, col_indices)])
    #signs[signs == 0] = 1
    signs = 2*(1*signs - 0.5)
    # if i == 8:
    #     print(list(zip(row_indices, col_indices)))
    #     print(row_indices)
    #     print(col_indices)
    #     print(cost_matrix)
    #     print(cost_matrix[1,2] < 1)
    #     print(signs) 
    #     print('----------')
    #     input('?!?!')
    # Create a list of matching column pairs = list of tuples
    matching_columns = list(zip(row_indices, col_indices))

    return matching_columns, signs #, cost_matrix

    

def find_best_A(lists_of_candidates, metric = 'MSE', assume_real = True, thres_real = 0.001):
    # calculate cross similarities for the highest K
    # print(len(lists_of_candidates))

    A_last = lists_of_candidates[-1]
    """
    the matrices below are num of matrices in last element X similarities
    """
    smallest_MSEs = np.zeros((len(A_last), len(lists_of_candidates)-1))*np.nan
    argmins_MSEs =  np.zeros((len(A_last), len(lists_of_candidates)-1))*np.nan
    at_least_one = False
    for A_num, A in enumerate(A_last):  
        if assume_real and ((A - np.real(A))**2).mean() > thres_real:
            smallest_MSEs[A_num, :] = np.inf
            argmins_MSEs[A_num, :] = -1
        else:     
            at_least_one = True
            for order_num, order in enumerate(lists_of_candidates[:-1]):
                """
                look only for reals?
                """                 
                candidates_MSEs = []
                found_in_order = False
                for candidate in order:
                    if assume_real and ((candidate - np.real(candidate))**2).mean() > thres_real:
                        pass
                    else:
                        found_in_order = True
                        if metric.lower() == 'mse':
                            A_MSE = np.mean((A - candidate)**2)
                        else:
                            raise ValueError('metric undefined yet')
                        candidates_MSEs.append(A_MSE)
                if not found_in_order:
                    smallest_MSEs[:, order_num] = np.inf
                    argmins_MSEs[:, order_num] = -1
                        
                argmin = np.argmin(candidates_MSEs)
                MSE =  candidates_MSEs[argmin]
                smallest_MSEs[A_num, order_num] = MSE
                argmins_MSEs[A_num, order_num] = argmin
                
    if not at_least_one:
        raise ValueError('not even at_least_one')            
    argmin_A_last = np.argmin(np.nansum(smallest_MSEs, 1))
    print('argmins_MSEs')
    print(argmins_MSEs)
    print('???????????????????')
    print(smallest_MSEs)
    print('-------------')
    return argmin_A_last, argmins_MSEs[argmin_A_last,:].flatten()
    
def find_best_A_consider_high_order(data, A_list, order_consider = 10, offset_list = []):
    if order_consider == -1:
        order_consider = data.shape[1]
        pred_K = [propagate_dyn_based_on_operator(data[:,0], A_i, order_consider-1, offset = offset_list[i])
        for i,A_i in enumerate(A_list)]

    else:
        pred_K = [k_step_prediction(data, A_i, order_consider, store_mid = False, t = -1, offset = offset_list[i]) for i,A_i in enumerate(A_list)]
    er_pred_k = [np.mean((data - pred_K_i)**2) for pred_K_i in pred_K]
    return np.argmin(er_pred_k)
    

def check_different_order(data, As, orders_to_check, include_full = True, offset = [], axs = [], fig = [], 
                          params_plot = {}, save_path = '.', save_formats = ['.png','.svg'], to_plot = True, to_save = True):
    # check increasing orders
    if not isinstance(orders_to_check, (tuple, list, np.ndarray)):
        if orders_to_check == int(orders_to_check):
            orders_to_check = np.arange(orders_to_check)
        else:
            raise ValueError('orders_to_check must be list/tuple/array or integer!')
            
    recos = [k_step_prediction_linear(data, As, k_i, store_mid = False, t = -1, offset = offset) for k_i in orders_to_check]
    
    if include_full:
        recos.append(propagate_dyn_based_on_operator(data[:,0], As, max_t=data.shape[1]-1, offset = offset ))
    if to_plot:        
        if checkEmptyList(axs):
            each_dir = int(np.sqrt(len(recos))) + 1
            fig, axs = create_3d_ax(each_dir, each_dir, figsize = (each_dir*5, each_dir*5))
            axs = axs.flatten()
        [plot_3d(data, ax = axs[i], params_plot =  {'color' : 'black', 'alpha' : 0.2, 'lw':1}) for i,reco in enumerate(recos)]    
        [plot_3d(reco, ax = axs[i], params_plot = params_plot) for i,reco in enumerate(recos)]
        
        [remove_background(ax) for ax in axs[len(recos):]]
        
        if to_save:
            fig.suptitle(save_path + os.sep + 'reco_orders' + save_formats[0])
            [fig.savefig(save_path + os.sep + 'reco_orders' + save_format) for save_format in save_formats]
    return recos
            
    
    
    
    
def check_applied_vs_predict_order(data, max_applied, max_predict, to_plot = True, to_save = True
                                   , offset = [], axs = [], fig = [], params_plot = {}, save_path = '.', save_formats = ['.png','.svg']):
    if not isinstance(max_applied, (tuple, list, np.ndarray)):
        if max_applied == int(max_applied):
            max_applied = np.arange(1, max_applied + 1)
        else:
            raise ValueError('orders_to_check must be list/tuple/array or integer! (applied)')
            
    if not isinstance(max_predict, (tuple, list, np.ndarray)):
        if max_predict == int(max_predict):
            max_predict = np.arange(1, max_predict + 1)

        else:
            raise ValueError('orders_to_check must be list/tuple/array or integer!')
            
            
      
    dim = data.shape[0]
    if to_plot:
        k_i = len(max_applied)
        k_p_i = len(max_predict)
        fig, axs = plt.subplots(k_p_i, k_i, figsize = (k_i*5,k_p_i*5), sharey = True)      
    recos = {}
    ers = {}
    recos_list = []
    for counter_k_i, k_i in enumerate(max_applied):        
        #fig, axs = plt.subplots(len(max_predict), k_i**dim, figsize = (k_i**dim,))      
        recos[k_i] = {}
        ers[k_i] = {}
        y_plus = data[:, k_i:]  
        y_minus = data[:, :-k_i]
        y_minus_expanded = np.vstack([y_minus, np.ones((1, y_minus.shape[1] )) ])
        
        # B here should be dim X T
        B =  infer_A_under_constraint(y_plus, y_minus_expanded, constraint = [], w_reg = {}, params = {},
                                      reeval = False , is1d_dir = 0, A_former = [], t = 0)

        B_k_part = B[:, :-1]

        
        """
        find all possible solutnios for A
        """
        if k_i > 1:
            # THIS IS A **LIST** OF OPTIONS
            A_hat = fractional_matrix_power_all_sols(B_k_part, k_i)            
        else:
            A_hat = [B_k_part]
            
         
        
        """
        offset
        the second part it (\sum_(A)^k)_(k=0)^K @ b
        result_power = np.linalg.matrix_power(A, k)
        """
        orders_list.append([])
        for sol_num, A_hat_i in enumerate(A_hat):
            #counter += 1
            sum_A_power = np.sum(np.dstack([ np.linalg.matrix_power(A_hat_i, k) 
                                            for k in range(0,k_i)])
                             ,2) 
            b_hat = np.linalg.pinv(sum_A_power) @ B[:,-1].reshape((-1,1))
            
            #print(b_hat.shape)
            #print('------------')
            orders_reco = check_different_order(data, A_hat_i, max_predict, include_full = True, offset = b_hat, axs = [], 
                              fig = [], params_plot = params_plot, save_path = save_path, save_formats = ['.png','.svg'], to_plot = False)
            recos[k_i][sol_num] = orders_reco
            recos_list[-1].extend(orders_reco)
            #print([len(reco_i[0]) for reco_i in orders_reco])
            ers_local = [((reco_i - data)**2).mean() for reco_i in orders_reco]
            
            ers[k_i][sol_num] = {j:ers_local[count] for count, j in enumerate(max_predict)}
        if to_plot:
            for count, j in enumerate(max_predict):
                list_for_predict_and_order = [np.real(ers[k_i][sol_num][j]) for sol_num in list(ers[k_i].keys())]
                #print(list_for_predict_and_order)                          
              
                #sns.histplot(list_for_predict_and_order, ax = axs[count,counter_k_i], width = 1)
                
                sns.heatmap(np.array(list_for_predict_and_order).reshape((1,-1)), ax = axs[count,counter_k_i], annot = True)
                axs[count,counter_k_i].set_title('applied %d, predict %d'%(k_i, j))
    if to_plot:
        #[ax.set_xscale('log') for ax in axs.flatten()]
        [ax.set_xticks([]) for ax in axs.flatten()]
        fig.tight_layout()
        fig.suptitle(save_path + os.sep + 'reco_orders' + save_formats[0])
        [fig.savefig(save_path + os.sep + 'reco_orders' + save_format) for save_format in save_formats]
        
        # plot together
        max_reco_len = np.max([len(val) for val in order_list])
        mat = np.zeros((len(order_list) , max_reco_len))
        for row, order_list_i in enumerate(order_list):
            mat[row,:len(order_list_i)] = order_list_i
        fig, ax  = plt.subplots()
        sns.heatmap(mat, ax = ax)
    return recos
    


def conj_transpose(v):
    return v.conj().T

def cosine_sim_complex_vectors(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError('v1 and v2 should have the same shape')
    if not is_1d(v1):
        list_vecs_v1 = [v1[:,i] for i in range(v1.shape[1])]
        list_vecs_v2 =  [v2[:,i] for i in range(v2.shape[1])]
        #print(cosine_sim_complex_vectors(list_vecs_v1[0], list_vecs_v2[0]))
        #print('ok?')
        return np.vstack([[cosine_sim_complex_vectors(v1_i, v2_i) for v1_i in list_vecs_v1] for v2_i in list_vecs_v2])
    else:
        v1_conj = conj_transpose(v1)
        v1_l2_norm = np.linalg.norm(v1)
        v2_l2_norm = np.linalg.norm(v2)
        cos_sim = np.dot( v1_conj, v2)/(v1_l2_norm * v2_l2_norm)
        return np.real(cos_sim)#np.abs(np.pi - np.real(cos_sim))
    
def train_linear_system_eigenvalas(data, K, Bs_main = [], constraint = [], w_reg = 3, params = {},
                             A_former = [],  w_offset = True , with_identity = False
                              ):
    if len(constraint) > 0:
        raise ValueError('future ext.')
        
    if checkEmptyList(Bs_main):
        _, Bs_main = find_Bs_for_dynamics(data, K)
    
    """
    find evals
    """
    _, v = np.linalg.eig(np.sum(np.dstack(Bs_main),2))
    #herehere
    return  optimize_A_using_optimizer(Bs_main, A0 = Bs_main[0], with_identity =  with_identity )

def infer_higher_order_b_given_A(data, k, A_hat, return_b_hat = True):
    # calculate the right side which include multiple powers of A
    data_minus = data[:,:-k]
    data_plus = data[:, k:]

    # so data diff is our estimation for y - A^K y_(t-k) = \sum A^k b = data_diff
    data_diff = data_plus - np.linalg.matrix_power(A_hat, k) @ data_minus
    A_hat_sum = np.sum(np.dstack([ 
        np.linalg.matrix_power(A_hat, k_i) for k_i in range(k) 
        ] ),2)
   
    if return_b_hat:
        b_hat = np.linalg.pinv(A_hat_sum) @ data_diff
        return b_hat, A_hat_sum, data_diff
    return A_hat_sum, data_diff
    
    
def find_A_basis_from_operators(operators, metric = 'mse', thres_percentile = 40, with_avg = True, wind_avg = 3, clustering_app = 'kmeans',
                                num_clusters = 3, to_plot = True, axs = [], fig = [], thres_corr_soft = 0.52, each = 3
                                , include_reorder = True, reorder_ref = [], params_plot = {}, percentile = True, thres_null = 0.8,
                                starting_points = []):
    
    if to_plot:
        if checkEmptyList(axs):
            fig, axs = plt.subplots(1, 5, figsize = (20,5))
    if isinstance(operators, list):
        identified_As_dstack = np.dstack(operators)
        operators = np.dstack(operators)     
        
    if len(operators.shape) == 3:
        identified_As_dstack = operators
        Ad2 = d3tod32(np.dstack(operators)) # tjos of samples X features
    elif 'identified_As_dstack' not in locals():
        identified_As_dstack = np.dstack([operators[i].reshape((each,each)) for i in range(operators.shape[0])])
    Ad2 = operators.T
    
    if metric == 'mse':        
        mat_er = np.vstack([[np.mean((Ad2[i] - Ad2[j])**2) for i in range(Ad2.shape[0])] for j in range(Ad2.shape[0])]).astype(float)
        if percentile:
            thres_er = np.percentile(mat_er, thres_percentile)
        else:
            thres_er = thres_percentile
        corr_mat = mat_er < thres_er #np.corrcoef(Ad2)
        if to_plot:
            sns.heatmap(mat_er, ax = axs[0])
            sns.heatmap(corr_mat, ax = axs[1])

        mat_corrs_thres = 1*(corr_mat).astype(float)
        print(mat_corrs_thres.shape)
        print('!!!!!!!!!!!!!!!')
    elif metric == 'corr':
        mat_er  = np.corrcoef(Ad2).astype(float)
        #thres_er = np.percentile(mat_er, thres_percentile)
        if percentile:
            thres_er = np.percentile(mat_er, thres_percentile)
        else:
            thres_er = thres_percentile
        corr_mat = mat_er > thres_er #np.corrcoef(Ad2)
        if to_plot:
            sns.heatmap(mat_er, ax = axs[0], **params_plot.get('heat1'))  
            sns.heatmap(corr_mat, ax = axs[1], **params_plot.get('heat2'))  

        mat_corrs_thres = 1*(corr_mat).astype(float)
    else:
        raise ValueError('metric undefined')
            
    # Apply k-NN
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(mat_corrs_thres) + 1
    mat_corrs_thres = labels.reshape((1,-1))*(corr_mat > thres_er).astype(float)
        
    if to_plot:
        if checkEmptyList(starting_points):
            starting_points = np.arange(len(labels)) + 0.5
        axs[2].plot( starting_points,labels, **params_plot.get('labels1'))  
        
        
    
    for label in np.unique(labels):

        locs_labels = np.where(labels == label)[0]
        #print('locs labels')
        #print(locs_labels)
        labels_part = labels[locs_labels]
        cur_group = corr_mat[labels == label,:][:,labels == label] 
        mean_group = cur_group.mean(0)
        #print('label')
        #print(label)
        #print('mean group')
        #print(mean_group )
        labels_part[mean_group < thres_corr_soft] = 0
        labels[locs_labels] = labels_part.flatten()
        if to_plot:
            axs[3].plot(starting_points.flatten(),labels.flatten(), marker = 'o', color = 'blue' ) # **params_plot.get('labels2'))  
    """
    add zeros in between
    """
    if with_avg:
        avg_labels = mov_avg(1*(labels != 0).reshape((1,-1)), wind = 3)
        labels[avg_labels.flatten() < thres_null]  = 0
        #labels[avg_labels.flatten() >= thres_null]  = 
        
        if to_plot:
            axs[4].plot(starting_points,labels, marker = 'o', **params_plot.get('labels2'))  
    un_labels = np.arange(1, num_clusters + 1)
    As = [np.nanmean(identified_As_dstack[:,:,labels == label],2) for label in un_labels if label != 0]
    As_dstack = np.dstack(As)
    """
    match A
    """
    if include_reorder:
        if checkEmptyList(reorder_ref):
            raise ValueError('you must provide reorder reference!')
            
        cost_matrix = np.vstack([[((A_i - F_i)**2).mean() for A_i in As] for F_i in reorder_ref ])#np.linalg.norm(A1[:, np.newaxis, :] - A2[:, :, np.newaxis], axis=0)
        
        # Solve the linear sum assignment problem
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            As_dstack_ordered = As_dstack[:,:,col_indices]
            labels_replace_dict = {**{col+1:row+1 for col, row in zip(col_indices, row_indices)},**{0:0}}
            labels = np.array([labels_replace_dict[val] for val in labels])
        except:
            As_dstack_ordered = As_dstack
        
        return labels, As, As_dstack, As_dstack_ordered ,  mat_er, corr_mat
        
    
    return labels, As, As_dstack, mat_er, corr_mat
    
        
from sklearn.cluster import KMeans    

def train_linear_system_opt_A(data, K, Bs_main = [], constraint = [], w_reg = 3, params = {},
                             A_former = [],  w_offset = True , weights = [], cal_offset = False,
                             infer_b_way = 'after_A', K_b = 20, w_b = [], with_identity = False):
                          
    if len(constraint) > 0:
        raise ValueError('future ext.')
        
    if checkEmptyList(Bs_main):
        Bs, Bs_main = find_Bs_for_dynamics(data, K, w_offset = w_offset)
    #print(len(weights))
    #print('############################')
    A_hat = optimize_A_using_optimizer(Bs_main, A0 = Bs_main[0], weights=weights, with_identity=with_identity)
    if cal_offset:
        dim = Bs_main[0].shape[0]
        if checkEmptyList(w_b):
            if not checkEmptyList(weights):
                w_b = weights.copy()
            else:
                w_b = np.ones(len(Bs_main))
                
        if infer_b_way == 'after_A':
            offsets_main = [Bs_i[:,-1].reshape((-1,1)) for Bs_i in Bs]
            A_powers = Bs_main.copy()
            A_powers_summation = [np.sum(np.dstack(A_powers[:i]),2) for i in range(1, len(A_powers)+1)]

            offset_hat = (np.linalg.pinv(np.vstack(A_powers_summation)*np.repeat(weights, dim).reshape((-1,1)))) @ (np.vstack(offsets_main)*np.repeat(weights, dim).reshape((-1,1)))

        elif infer_b_way == 'each':
            A_hat_sum = []
            data_diff = []
            b_hats = []
            for k in range(1, K_b + 1):
                b_hat_i, A_hat_sum_i, data_diff_i = infer_higher_order_b_given_A(data, k, A_hat, return_b_hat = True)
                #print(A_hat_sum_i.shape)
                #print(data_diff_i.shape)
                A_hat_sum.append(w_b[k-1]*A_hat_sum_i)
                data_diff.append((w_b[k-1]*data_diff_i).mean(1).reshape((-1,1)))
                b_hats.append(b_hat_i)
            offset_hat = np.linalg.pinv(np.vstack(A_hat_sum)) @ np.vstack(data_diff)

        else:
            raise ValueError('infer b way underfined')
        
        return A_hat, offset_hat, b_hats
    return A_hat









    
def train_linear_system(data, K, constraint = [], w_reg = 3, params = {},
                              reeval = True , is1d_dir = 0, A_former = [], t = 0, 
                              reweigh_As_func = np.nansum, w_offset = True, 
                              method_find_b = 'final', sigma1 = 0.001, sigma2 = 1.1, 
                              to_save = True, to_plot = True,
                              path_save = '.', way_to_filter_A = 'by_higher_order', 
                              params_way_filter_A = {'order_consider': 480},addi = ''):
    """
    fractional_matrix_power_all_sols(B, n):
    Parameters
    ----------
    data : data dim X time    
    K : max reconstruction order. scalar integer
    method_find_b - can be 'each_order' - finds b based on local estimation of A
                  - can be 'final'   - last step
                  - order 0 
                  - final is easierst
                  
    offsets - list  of lists fir all orders
    find_best_A_consider_high_order(data, A_list, order_consider = 10, offset = []):
    
    way_to_filter_A options:    
        - by_sub_group_with_last:
        - by_higher_order:
    """
    A_hats = []
    offsets = []
    B_non_k_parts = []
    dim = data.shape[0]
    es = []
    es_look = []
    reco_each = []
    cur_each_look = []
    ks = np.arange(1, K+1)
    all_looks = []
    lists_of_candidates = []
    
    
    #all_As = []
    if w_offset:
        shifts_len = data.shape[0]
        
    counter = 0
    if to_plot:
        dim = data.shape[0]
        num_sols = np.sum([k**dim for k in range(1, K+1)])
        each_dim = int(np.sqrt(num_sols)) + 1
        fig, axs = create_3d_ax(each_dim,each_dim,figsize = (each_dim*5,each_dim*5))
        axs = axs.flatten()
        
        
        each_dim = int(np.sqrt(K)) + 1
        fig2, axs2 = create_3d_ax(each_dim, each_dim, figsize = (each_dim*5,each_dim*5*3))
        axs2 = axs2.flatten()
        
        
        fig_check_look, axs_check_look = create_3d_ax(each_dim, each_dim, figsize = (each_dim*5,3*each_dim*5))
        axs_check_look  = axs_check_look.flatten()
        
    for k_i in ks:        
        counter += 1
        if not w_offset:
            # find B
            y_plus = data[:, k_i:]  
            y_minus = data[:, :-k_i]
            B = infer_A_under_constraint(y_plus, y_minus, constraint, w_reg, params,
                                          reeval , is1d_dir, A_former = A_former, t = 0)
            if k_i > 1:
                A_hat = find_A_fractional(k_i, B)
            else:
                A_hat = B
            A_hats.append(A_hat)    
            
        else:
            print('here')
            y_plus = data[:, k_i:]  
            y_minus = data[:, :-k_i]
            y_minus_expanded = np.vstack([y_minus, np.ones((1, y_minus.shape[1] )) ])
            
            # B here should be dim X T
            B =  infer_A_under_constraint(y_plus, y_minus_expanded, constraint, w_reg, params,
                                          reeval , is1d_dir, A_former = A_former, t = 0)

            B_k_part = B[:, :dim]
            B_offset_part = B[:,dim:]
            B_non_k_parts.append(B_offset_part)  
            
            """
            find all possible solutnios for A
            """
            if k_i > 1:
                # THIS IS A **LIST** OF OPTIONS
                A_hat = fractional_matrix_power_all_sols(B_k_part, k_i)            
            else:
                A_hat = [B_k_part]
                
            A_hats.append(A_hat)    
            
            """
            offset
            the second part it (\sum_(A)^k)_(k=0)^K @ b
            result_power = np.linalg.matrix_power(A, k)
            """
            offsets.append([])
            es_look.append([])
            for A_hat_i in A_hat:
                counter += 1
                sum_A_power = np.sum(np.dstack([ np.linalg.matrix_power(A_hat_i, k) 
                                                for k in range(0,K)])
                                 ,2) 
                b_hat = np.linalg.pinv(sum_A_power) @ B_offset_part
                offsets[-1].append(b_hat.reshape((-1,1)))                      
                
                
                reco_look = propagate_dyn_based_on_operator(data[:,0], A_hat_i, data.shape[1] - 1, offset = b_hat)
                
                if to_plot:
                    plot_3d(reco_look, ax = axs[counter],  params_plot = {'lw':0.2, 'alpha':0.6})
                    
                    
                    axs[counter].set_title(k_i, fontsize = 10)
                
                cur_each_look.append(reco_look)                    
       
                e_look = np.nanmean((data - reco_look)**2)    
                es_look[-1].append(e_look)
                
            reco_k = B @ y_minus_expanded
            
            if to_plot:
                plot_3d(reco_k, ax = axs2[k_i-1],  params_plot = {'lw':6, 'alpha':0.9})
                axs2[k_i - 1].set_title('Reci Order for the same k = %d'%k_i, fontsize = 20)
                
            reco_each.append(reco_k)
            e = np.nanmean((y_plus - reco_k)**2)
            es.append(e)  
            all_looks.append(cur_each_look)
            
            
    """
    SELECT BEST A
    find best A given all candidates
    argmin _others is the best A for each future order
    """
    if way_to_filter_A == 'by_sub_group_with_last':
        argmin_A_last, argmin_others = find_best_A(A_hats, metric = 'MSE')
        
        """
        
        calculate errors for new identified best matrx
        """
        A_last = A_hats[-1][argmin_A_last]
        offset_last = offsets[-1][argmin_A_last]
        
        """
        es_look is a list of lists of errors for the lookahead of orders.
        """
        es_last = es_look[-1][argmin_A_last]
        argmin_others = np.append(argmin_others, argmin_A_last).astype(int)
        print('================================')
        
    elif way_to_filter_A == 'by_higher_order':
        argmin_others = []
        for c, A_sublist in enumerate(A_hats):
            if 'order_consider' not in params_way_filter_A:
                raise ValueError('You must provide max order to consider if way_to_filter_A is by_higher_order')
            min_arga_for_order = find_best_A_consider_high_order(data, A_sublist,  offset_list = offsets[c],
                                                                 **params_way_filter_A )
            argmin_others.append(min_arga_for_order)
            look_best = all_looks[c][min_arga_for_order]
            #print(look_best.shape)
            #print('-----------')
            plot_3d(look_best, ax = axs_check_look[c])
            axs_check_look[c].set_title('order %s'%str(c))
    else:
        raise ValueError('way_to_filter_A undefinged (is %s)'%way_to_filter_A )
        
     
    #print(argmin_others)
    #print(len(argmin_others))
    #display(argmin_others)
    
    #print([argmin_others[i] for i, A_list in enumerate(A_hats)])
    other_As = [np.real(A_list[argmin_others[i]] )
                for i, A_list in enumerate(A_hats)] #+ [A_last]
    other_offsets = [np.real(offset[argmin_others[i]] )
                for i, offset in enumerate(offsets)] #+ [offset_last]
    #print(len(es_look))
    es_look_new = np.array([e[argmin_others[i]] for i, e in enumerate(es_look)]) #+ [es_last])
    
    """
    A_hats
    """
    # reco
    A_hats_dstack = np.dstack(other_As) 
    """
    weights
    """
    es_inv = 1/np.array(es_look_new) 
    
    weights = find_weight(ks, es_inv, sigma1 = sigma1, sigma2 =sigma2, norm_w = True, norm_direction = -1)
    weights /= weights.sum()
    

    A_avg = np.nansum(weights.reshape((1,1,-1)) * A_hats_dstack, 2)
    

    if method_find_b == 'final':
        # solves y_next = A_avg @ y_min + b
        offset = np.mean(data[:, 1:] - A_avg @ data[:, :-1], 1)        
    elif method_find_b ==  'each_order':
        offset = np.nanmean(np.hstack(offsets) * weights.reshape((1,-1)), 1)
    else:
        raise ValueError('not yet implemented')        
    
    cur_reco = propagate_dyn_based_on_operator(data[:,0], np.dstack([A_avg]*data.shape[1]),data.shape[1], offset = offset)
    
    if to_plot:
        [remove_edges(ax_i) for ax_i in axs]
        [remove_background(ax_i) for ax_i in axs]
        fig.tight_layout(h_pad=-0.5, w_pad = -0.5)
        
    if to_save and to_plot:
        addi = '%s_max_K_%d'%(addi, K)
        fig.suptitle(path_save + os.sep + 'linear_different_recos_%s.png'%addi)
        fig.savefig(path_save + os.sep + 'linear_different_recos_%s.png'%addi, transparent = True, bbox_inches='tight')
        fig.savefig(path_save + os.sep + 'linear_different_recos_%s.png'%addi, transparent = True, bbox_inches='tight')
        
        fig2.suptitle(path_save + os.sep + 'linear_different_recos%s_2.png'%addi)
        fig2.savefig(path_save + os.sep + 'linear_different_recos_%s_2.png'%addi, transparent = True, bbox_inches='tight')
        fig2.savefig(path_save + os.sep + 'linear_different_recos_%s_2.png'%addi, transparent = True, bbox_inches='tight')  
        
        fig_check_look.suptitle('Recos for best A under check_order \n %s'%path_save + os.sep + 'K_recos_%s_predict_%s.png'%(addi,params_way_filter_A['order_consider']))
        fig_check_look.savefig(path_save + os.sep + 'K_recos_%s_predict_%s.png'%(addi,params_way_filter_A['order_consider']),
                               transparent = True, bbox_inches='tight')
        fig_check_look.savefig(path_save + os.sep + 'K_recos_%s_predict_%s.png'%(addi,params_way_filter_A['order_consider']), 
                               transparent = True, bbox_inches='tight')          
    #fig.tight_layout()
    return other_As, A_hats, A_avg, cur_reco, reco_each, all_looks, offset, other_offsets, weights, es, es_look
    
    
def check_1d(mat):        
    return np.max(mat.shape) == len(mat.flatten())    
    
    
        
def create_reco_new(x,  coefficients, F, type_reco = 'lookahead', plus_one = 0,
                    thres_max = 40, thres_min = 5.5,  seed = 0, offset = [], with_identity = False, max_try = 500):
    if type_reco == 'step' and x.shape[1]-1 != coefficients.shape[1]:
        raise ValueError('mismatch in dim. x dim is %d while c dim is %d'%( x.shape[1], coefficients.shape[1]))
    if checkEmptyList(x) and type_reco == 'lookahead':
        np.random.seed(seed)
        x = np.random.rand(F[0].shape[0]).reshape((-1,1))        
        
    T = coefficients.shape[1] + 1
    if not check_1d(x) and type_reco == 'lookahead':
        x = x[:,0]
        x = np.array(x).reshape((-1,1))
    elif  type_reco == 'lookahead':
        
        x = np.array(x).reshape((-1,1))
        print('x0 ok')
        print(x)
        
    #t = 0
    # print('coefficients max')
    # print(np.max(coefficients))
    # print(np.max(x))
    # print([np.max(np.abs(f)) for f in F])
    if type_reco == 'lookahead':
        if checkEmptyList(offset):
            for t in range(T-1):
                x = np.hstack([x,   (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @  x[:,-1] ).reshape((-1,1)) ])
                #if np.isnan(np.max(x)):
                    
                # print('coefficients max')
                # print(np.max(coefficients))
                # print(np.max(x))
                # print(x[:,-2:])
                # print([np.max(np.abs(f)) for f in F])
                # print(t)
                # print('8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888')
                # input('wth')
                x[np.abs(x)>max_try] = max_try
        else:
            for t in range(T-1):
                x = np.hstack([x,   (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @  x[:,-1] ).reshape((-1,1)) + offset.reshape((-1,1))])
                x[np.abs(x)>max_try] = max_try
            
        
        
    else:
        #print(len(F))
        if checkEmptyList(offset):
            x_hat = np.hstack([
                (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @   x[:,t].reshape((-1,1))).reshape((-1,1))
                for t in range(x.shape[1]-1)])
        else:
            x_hat = np.hstack([
                (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @   x[:,t].reshape((-1,1))).reshape((-1,1)) + offset.reshape((-1,1))
                for t in range(x.shape[1]-1)])
        x = np.hstack([x[:,0].reshape((-1,1)), x_hat]) 
        
    return x  
    
from scipy.interpolate import interp1d        
def find_start_end_points(labels, starting_points, T):
    #print('------------------------------')
    #print(len(labels))
    #print(len(starting_points))
    # labels_shift = np.array([1] + list(1*(labels != 0)) + [1] )
    # start_points = starting_points[np.diff(labels_shift)[:-1] == -1]
    # end_points = starting_points[np.where(np.diff(labels_shift) == 1)[0] + 1]
    # inverse_start = np.array([0] + list(end_points+1))
    # inverse_end = np.array( list(start_points+1) + [T] )
    np.random.seed(0)
    starting_points = np.arange(65)
    labels = 1*(np.random.rand(65) < 0.1)
    labels_shift = np.array([1] + list(1*(labels != 0)) + [1] )
    start_points = starting_points[np.diff(labels_shift)[:-1] == -1]
    end_points = starting_points[np.where(np.diff(labels_shift[1:]) == 1)[0] ]
    
    inverse_start = np.array([0] +list(np.minimum(end_points, T)))
    inverse_end = np.array( list(start_points+1) + [T] )
    return np.hstack([start_points.reshape((-1,1)), end_points.reshape((-1,1))]), np.hstack([inverse_start.reshape((-1,1)), inverse_end.reshape((-1,1))])
        
    # THE SAME AS FIND C
def find_A_decomposed_K_steps(cur_data, F, K, sigma1, sigma2, A_former = [],  w_reg = {}, params = {}, 
                              reeval = True, reweigh_As_func = np.nansum, norm_w = True, smooth_w = 0 , lasso_params = {},
                              l1_w = 0, c_start = []):
    """
    Decompose time series data into matrices A, coefficients cs, and reconstructions over K steps.
    
    Parameters:
    - cur_data (ndarray): Input data of shape (dim, T).
    - F: An unspecified parameter (not detailed in this context).
    - K (int): Number of decomposition steps.
    - sigma1 (float): Parameter for finding weights in the reweighting process.
    - sigma2 (float): Another parameter for finding weights.
    - A_former (list of ndarrays): Previous matrices A for smoothness (not fully implemented).
    - w_reg (dict): Weight regularization (not detailed in this context).
    - params (dict): Additional parameters (not detailed in this context).
    - reeval (bool): Boolean indicating whether to reevaluate during decomposition.
    - reweigh_As_func (function): Function for reweighing As during the reweighting process.
    - norm_w (bool): Boolean indicating whether to normalize weights.
    
    Returns:
    - A_cur_rewighs (ndarray): Reweighted matrices A after K steps of decomposition.
    - cs_rewighs (ndarray): Reweighted coefficients cs.
    - cs_dstack (ndarray): Stacked coefficients cs.
    - As_stores (list of ndarrays): List of matrices A for different orders during decomposition.
    - weights (ndarray): Weight matrix used in the reweighting process.
    
    Operation:
    The function iteratively decomposes the input time series data `cur_data` over `K` steps. At each step,
    it finds matrices A, coefficients cs, and reconstructions. It then estimates the best A by re-weighting the
    reconstructions based on calculated weights. The process involves calculating errors, finding weights,
    and re-weighting As and cs accordingly. The results are stored and returned for further analysis.
    
    # infer the data in the Ks time point using 
    # cur data is dim X time local
    """
    

    if not checkEmptyList(A_former) and 'smooth' in constraint:
        raise ValueError('future extension! smoothness in decomposition')
    else:
        
        As_stores = []
        cs_stores = []
        stores_reco = []
        
        K = np.min([K, cur_data.shape[1]])       
        T = cur_data.shape[1]
        cur_data_k_plus = cur_data.copy()[:,1:]
        cur_data_k_minus = cur_data.copy()[:,:-1]
        for k in range(K):     
            
            # here A is a 3d mat of dim X dim X t;             # cs is a list of duration t each M X 1
            A, cs, cur_reco = find_A_decomposed_1_step_period(cur_data_k_plus, cur_data_k_minus, F, A_former, 
                                                              w_reg, params, reeval = reeval, smooth_w = smooth_w, 
                                                              lasso_params = lasso_params , l1_w = l1_w,  c_start =  c_start)
            
            As_stores.append(A)
            if k > 0:
                cs = np.hstack([cs_former[:,:k], cs])
            cs_former = cs.copy()
            cs_stores.append(cs)
            if is_1d(cur_reco):
                cur_reco = cur_reco.reshape((-1,1))
            stores_reco.append(cur_reco)
            
            
            cur_data_k_plus = cur_data_k_plus[:, 1:]
            cur_data_k_minus = cur_reco[:, :-1]
        full_recos = [create_reco_new(cur_data, cs_stores[i],  F, type_reco='lookahead')[:,:-1] for i  in range(K)]    
        """
        estimate best A by re-weighting
        (stores_reco is a list of K elements, each with T - 1 - k elements. 
        """
        if K == 1:
            weights = [1]
            cs_dstack = np.expand_dims(cs_stores[0], 2)
            cs_rewighs = cs_dstack
            A_cur_rewighs = As_stores
            reco_look = full_recos[0]
            es = np.mean((reco_look - cur_data[:,:-1])**2)
            return A_cur_rewighs, cs_rewighs, cs_dstack, As_stores, weights, stores_reco, es, reco_look 
        # PAD DATA
        dim = cur_data.shape[0]
        recos = np.nan*np.zeros((dim, T - 1, K))
        cs_dstack = np.nan*np.zeros((dim, T - 1, K))
        weights = np.nan*np.zeros(( K, T)) # weights =
        for mat_count, mat in enumerate(stores_reco):
            recos[:,-mat.shape[1]: ,mat_count] = mat #[:,t].reshape((-1,1,k+1))
            cs_dstack[:, : ,mat_count] = cs_stores[mat_count]
            #cs_dstack[:, -mat.shape[1]: ,mat_count] = cs_stores[mat_count]
        real = cur_data[:, 1:].reshape((dim, -1,1))  
        
        es = np.nanmean((real - np.dstack(full_recos))**2, 0) #T by K 

        ks = np.repeat(np.arange(K).reshape((1,-1)), T-1, axis = 0)
        
        # a mat of T X K weights
        weights = find_weight(ks, es, sigma1 = sigma1, sigma2 = sigma2, norm_w = norm_w)
        weights[weights == 0] = np.nan
        A_cur_rewighs =[] # A_cur_rewigh
        cs_rewighs = []
       
        for t in range(1,T):
            """
            how many orders we have for this time point?!
            """
            K_max = np.min([t, len(As_stores)])        
            A_cur = np.dstack([As_stores[k][:,:,t-k-1] for k in range(K_max)])
            A_cur_rewigh = reweigh_As_func(np.dstack([A_cur * weights[t-1,:K_max].reshape((1,1,-1))]), 2)
            A_cur_rewighs.append(A_cur_rewigh)
            cs_rewighs.append(reweigh_As_func(np.dstack([cs_dstack[:,t-1,:K_max] * weights[t-1,:K_max].reshape((1,1,-1))]), 2).reshape((-1,1)))   
            
            
            
            
        A_cur_rewighs = np.dstack(A_cur_rewighs)
        cs_rewighs = np.hstack(cs_rewighs)
        reco_look = create_reco_new(cur_data, cs_rewighs,F)

        return A_cur_rewighs, cs_rewighs, cs_dstack, As_stores, weights, stores_reco, es, reco_look
        

    
 
                            
    
    
# """
# check lookahead

# """

# x, stores = k_step_prediction(data_between, As, K, store_mid = True, t = -1)

# """
# matrix of K x T
# """
# es = np.vstack([np.sum((x_i[:, t_start:t_end +1] - data[:, t_start:t_end +1])**2, 0) for  x_i in stores]) TO CHANGE TIME HERE

# """
# weight for each time point
# """
# w_t = find_weight(np.arange(1,K+1), es[:,1], sigma1 = sigma1, sigma2 = sigma2) 
# if norm_w:
#     w_t /= w_t.sum()

# """
# calculate different solutions dim X K X 2. The 1st 3d dim is the ground truth, the 2nd is the inference for the former time point
# """
# data_plus_data_minus = np.hstack([np.dstack(find_propagations(As, t, data, k, constraint = constraint, w_reg = w_reg, 
#                                                               shape_return = (-1, 1, 1))) 
#                                for k in range(1,K+1)])



# data_plus_data_minus_w = data_plus_data_minus  * w_t.reshape((1,-1,1))  

# """
# Infer_A_under_constraint
# """
# #infer_A_under_constraint(y_plus, y_minus, constraint = ['l0'], w_reg = 3, params = {}, reeval = True , is1d_dir = 0, A_former = [], t = 0)
# #print('data_plus_data_minus_w shape')
# #print(data_plus_data_minus_w.shape)
# if t > 1:
#     A_former = As[:,:,np.max([0,t-2])]
# else:
#     A_former = []






    


# def  infer_A_under_constraint(y_plus, y_minus, constraint = ['l0'], w_reg = 3, params = {},
#                               reeval = True , is1d_dir = 0, A_former = [], t = 0):
#      #future not l0   

#      #if type(constraint) != type(w_reg) and  (isinstance(constraint, list) and len(constraint) != 1) :
#      #    print('w_reg and cnostraing need to be of the same type. but %s, %s'%(str(constraint), str(w_reg)))
         

#      if checkEmptyList(A_former) and 'smooth' in constraint and t > 0:
#          raise ValueError('?!?!?!')
         
#      if is_1d(y_plus):
#          if is1d_dir == 0:
#              y_plus = y_plus.reshape((-1,1 ))
#          else:
#              y_plus = y_plus.reshape((1,-1 ))
            
             
#      if is_1d(y_minus):
#          if is1d_dir == 0:
#              y_minus = y_minus.reshape((-1,1 ))
#              shape_inv = (1,-1)
#          else:
#              y_minus = y_minus.reshape((1,-1 ))
#              shape_inv = (1,-1)
#      else:
#          shape_inv = y_minus.shape[::-1]
        
#      try:    
#          A_hat = y_plus @ np.linalg.pinv(y_minus)
#      except:
#          print('y plus ?!')
#          print(y_plus)
#          A_hat = y_plus @ np.linalg.pinv(y_minus + np.random.rand(*y_minus.shape)*0.01)
#      for  constraint_i in  constraint:
#          w_reg_i = w_reg[constraint_i]
#          A_hat = apply_constraint_after_order(y_plus, y_minus, constraint_i, w_reg_i, A_former = A_former, t = t, A_hat = A_hat, reeval = True)    
#      return A_hat
    
def apply_constraint_after_order(y_plus, y_minus, constraint_i, w_reg_i, A_former = [], t = 0, A_hat = [], reeval = True):
    #if 'smooth' in constraint and t > 0: # this is for a single time point
    if constraint_i == 'smooth' and t > 0:
        #smooth_w = w_reg['smooth']
        #print(y_plus.shape)
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        A_hat = find_operator_under_l2_or_smoothness( y_plus, y_minus, l2_w = 0, 
                                                     smooth_w = w_reg_i, t = t, A_minus = A_former, reeval = reeval, mask = A_hat != 0 )
   
        
        
    elif  constraint_i == 'l0':
        #raise ValueError('wth?!')
        w_reg_i = int(w_reg_i)
        if checkEmptyList(A_hat):
            raise ValueError('yopu must provide A_hat')
        A_hat = keep_thres_only(A_hat, direction = 'lower' , thres = w_reg_i, num = True, perc = False)    
        
        
        if reeval :
            A_hat = reevaluate_A_under_mask(y_plus, y_minus, A_hat != 0)
          
    return A_hat 


def find_A_based_on_multiple_orders_for_period(stored_3d_weighted, cur_data,  constraint , w_reg, params = {},
                              reeval = True , is1d_dir = 0, A_former = [], t = 0 ):
    #cur_data = cur_data.reshape((-1, 1, cur_data.shape[1]))
    # find A that maximize them together
    K = stored_3d_weighted.shape[1]
    cur_data_hstack = np.hstack([cur_data]*K)
    stored_stack = np.hstack([stored_3d_weighted[:,i,:] for i in range(K)])
    # infer A
    A = infer_A_under_constraint(y_plus, y_minus, constraint =  constraint, w_reg = w_reg, params = params,
                                  reeval = reeval , is1d_dir = is1d_dir, A_former = A_former, t = t)
    return A 
    
    

    

def find_propagations(As, t, data, k, constraint = 'l0', w_reg = 3, shape_return = (-1)):
    # t is time to update
    # As are all As
    # FUTURE - CAN MAKE THIS STEP MORE EFFICIENT BY USING THE STORED INFO
    min_a = np.max([0,t-k])
    data_0 = data[:, min_a]
    if k > 1 and t > 0:        
        data_prop_t_minus_1 =  propagate_dyn_based_on_operator(data_0, As[:,:,min_a: t-1], max_t = t-1 - min_a)[:,-1]
    if k == 0:
        data_prop_t_minus_1 = data[:,t]
    else:
        data_prop_t_minus_1 = data_0.copy()
        
    data_t = data[:,t]
    
    
    return data_t.reshape(shape_return), data_prop_t_minus_1.reshape(shape_return)
    

 
def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't')
    return str_to_change
    
    


def train_LOOKAHEAD(data, K_f = 10, As = [], constraint = ['l0', 'smooth'], w_reg = {}, max_error = 1, 
                    sigma1 = 7, sigma2 = 12, norm_w = True, reeval = True,
                    backprop = False, max_iter = 500,
                    seed = 0, store_As = True, is1d_dir = 0, params = {}, constraint_params_increase = {}, max_contraint  = {}, 
                    to_avg = True, avg_params = {'wind': 3, 'freq': 2}, given_periods = [] ):
    """
    Train the LOOKAHEAD model on input data.
    
    Parameters:
    - data (numpy.ndarray): Input data matrix.
    - K_f (int): Lookahead parameter.
    - As (list or numpy.ndarray): Initial values for the matrix As.
    - constraint (list): List of constraints to apply during training.
    - w_reg (dict): Dictionary of regularization weights.
    - max_error (float): Maximum error allowed for training termination.
    - sigma1 (float): Sigma parameter for weight calculation.
    - sigma2 (float): Sigma parameter for weight calculation.
    - norm_w (bool): Normalize weights if True.
    - reeval (bool): Reevaluate constraints during training if True.
    - backprop (bool): Use backpropagation (not implemented) if True.
    - max_iter (int): Maximum number of iterations.
    - seed (int): Random seed.
    - store_As (bool): Store As matrices during training if True.
    - is1d_dir (int): 1D direction parameter.
    - params (dict): Additional parameters for constraint inference.
    - constraint_params_increase (dict): Parameters for increasing constraints.
    - max_contraint (dict): Maximum constraints for regularization.
    - to_avg (bool): Apply moving average if True.
    - avg_params (dict): Parameters for moving average.
    - given_periods (list): List of periods for linear dynamics.
    
    Returns:
    - As (numpy.ndarray): Trained As matrices.
    - errors (list): List of errors at each iteration.
    - x (numpy.ndarray): Predicted output.
    - elapsed_time (float): Training elapsed time.
    - store_As_dict (dict): Dictionary storing As matrices at each iteration (if store_As is True).
    """
    # given periods define the periods of linear dynamis. this is a list of lists. each sublit define the initial index and end index of each period
    np.random.seed(seed)
    T = data.shape[1]
    for key in w_reg.keys():
        if key not in max_contraint :
            max_contraint[key] = np.inf
        if key not in constraint_params_increase:
            constraint_params_increase[key]= 1
    
    
    if checkEmptyList(As):
        As = np.random.rand(data.shape[0], data.shape[0], data.shape[1]-1)
    
    iter_num = 0
    
    """
    FIRST STEP
    """
    x, stores = k_step_prediction(data, As, K_f, store_mid = True, t = -1)
    error = np.sum((x - data)**2)
    errors = []
    print('error')
    print(error)
    errors.append(error)     
    
    
    start_time = time.time()
    
    
    if store_As:
        store_As_dict = {}
    
    while error > max_error and iter_num < max_iter:    
        if iter_num == 0:
            print('enter the loop!!!!!!!!!!!!')    
        print('iter num %d'%iter_num)
        
        if not checkEmptyList(given_periods):
                As = infer_A_under_constraint_under_period(data,
                                               constraint = constraint, 
                                               w_reg = w_reg, params = params,
                                               reeval = reeval, is1d_dir = is1d_dir,
                                               given_periods = given_periods)
                
        
        else: 
            for t in range(1,T):
    
                
                K = np.min([K_f, t])
                t_start = np.max([t - 1,0])         
                t_end = np.min([t + K - 1, As.shape[2]])
                """
                check lookahead
                """
                x, stores = k_step_prediction(data, As, K, store_mid = True, t = -1)
    
                """
                matrix of K x T
                """
                es = np.vstack([np.sum((x_i[:, t_start:t_end +1] - data[:, t_start:t_end +1])**2, 0) for  x_i in stores])
                
                """
                weight for each time point
                """
                print(es)
                print(len(es))
                print(len(stores))
                #print(stores)
                # print(es.shape)
                # print('shape above')
                print(' ')
                w_t = find_weight(np.arange(1,K+1), es[:,1], sigma1 = sigma1, sigma2 = sigma2, norm_w = norm_w, norm_direction = -1) 

                
                """
                calculate different solutions dim X K X 2. The 1st 3d dim is the ground truth, the 2nd is the inference for the former time point
                """
                data_plus_data_minus = np.hstack([np.dstack(find_propagations(As, t, data, k, constraint = constraint, w_reg = w_reg, 
                                                                              shape_return = (-1, 1, 1))) 
                                               for k in range(1,K+1)])
                
                
                
                data_plus_data_minus_w = data_plus_data_minus  * w_t.reshape((1,-1,1))  
                
                """
                Infer_A_under_constraint
                """
                #infer_A_under_constraint(y_plus, y_minus, constraint = ['l0'], w_reg = 3, params = {}, reeval = True , is1d_dir = 0, A_former = [], t = 0)
                #print('data_plus_data_minus_w shape')
                #print(data_plus_data_minus_w.shape)
                if t > 1:
                    A_former = As[:,:,np.max([0,t-2])]
                else:
                    A_former = []
                    
                A_t = infer_A_under_constraint(data_plus_data_minus_w[:,:, 0], 
                                               data_plus_data_minus_w[:,:,1],
                                               constraint = constraint, 
                                               w_reg = w_reg,
                                               params = params,
                                               reeval = reeval, is1d_dir = is1d_dir, A_former = A_former, t = t - 1, 
                                              )
                
                #constraint_params
                for key,val in constraint_params_increase.items():
                    if w_reg[key] <= max_contraint[key]:
                        w_reg[key] *= val
                    
                
                
                As[:,:,t-1] = A_t
              
                # if len(A_former) > 1:
                #     print(((A_t - A_former)**2).sum())
                #     print('!!!!!!!!!!!!!!!!!')
    
                    
                error = np.sum((x - data)**2)
                errors.append(error)
                
                print('error inside: %f, t = %d'%(error, t))
            
        """
        MOVING AVG
        """
        
        if to_avg:
            if  np.mod(iter_num,  avg_params['freq']) == 1:
                print('As.shape')
                print(As.shape)
                As = mov_avg(As, wind = avg_params['wind'], axis = 2 )
                
        if backprop:
            raise ValueError('future imp')
        if store_As:
            store_As_dict[iter_num] = As
        print('error %s'%error)
        
        errors.append(error)
        iter_num += 1
        
        # print(check_smoothness_operators(As))
        # print('?!?!?!!?!?!?!?!?!?')
        # input('ok?!?!?!?')
        
        
    end_time = time.time()
    

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time    
    if store_As:
        return As, errors, x, elapsed_time, store_As_dict
    
    return As, errors, x, elapsed_time
    
def choose_random_min_max(min_v, max_v, seed = 0):
    np.random.seed(seed)
    return np.random.rand()*(max_v - min_v) + min_v
    

def check_smoothness_operators(operators):
    return np.mean(np.diff(operators, 2)**2)
    
    
    
    
    
    
def find_local_operators(As, zs  = [], cs = []):
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
  

    
    
def find_lorenz_with_uncorr_x_and_y(xmin = -1, xmax = 1, ymin = -1, ymax = 5, zmin = -5, zmax = 5, dt_min = 0.0010, dt_max = 0.04, max_t_max = 10,max_t_min = 1,
                  sigma_min = 8, sigma_max = 12, beta_min = 4/3, beta_max = 10/3,  rho_min = 10, rho_max = 38, return_operators = True, option = 2, num_iters = 100,
                  seed = 0):
    params = []
    corrs = []
    start_time = time.time()
    for seed in range(num_iters):
        

        
        psi0 = [choose_random_min_max(xmin, xmax, seed),choose_random_min_max(ymin, ymax,seed),choose_random_min_max(zmin, zmax, seed) ]    
        rho = choose_random_min_max(rho_min, rho_max, seed)
        sigma = choose_random_min_max(sigma_min, sigma_max, seed)
        beta = choose_random_min_max(beta_min, beta_max, seed)
        dt =  choose_random_min_max(dt_min, dt_max, seed)
        max_t=  int(choose_random_min_max(max_t_min, max_t_max, seed)) #.astype(int)
        params_dict = {
        'psi0': psi0,
        'dt': dt,
        'max_t': max_t,
        'sigma': sigma,
        'beta': beta,
        'rho': rho,
        'return_operators': return_operators,
        'option': option
        }
        lorenz_mat1, _ = create_lorenz(psi0, max_t = max_t, option = 2, dt = dt)
        if np.max(np.abs(lorenz_mat1)) < 1000:
            
            cor = np.corrcoef(lorenz_mat1)[0,1]
            corrs.append(cor)
            params.append(params_dict)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time # in sec
    return corrs, params, elapsed_time
    


    
    
    
    
    
########################################################
# GENERAL PARAMETERS
########################################################
colors_els = np.array(['forestgreen','limegreen','lime', 'darkorange','peru','gold', 'purple','orchid','deeppink'])
ticklabels_size = 19
subtitle_size = 39
labels_size = 29
title_size = 29
share_axis = False
plot_params_basic = {'lw':3, 'alpha':0.8, 'marker':'.'}
plot_params = {'params_plot':plot_params_basic}
labelpad = 15
to_save = True
cmaps = {'real':'Greys','LINOCS':'Blues', '1 step':'Oranges', 'Observed':'Greys','DAD': 'Reds',
    'DAD $\ell_2$': 'Reds',
    'DAD reweigh': 'Reds',
    'reweigh $\ell_2$' : 'Reds'}

models = ['Observed', 'LINOCS', '1 step',  'DAD', 'DAD reweigh', 'reweigh $\ell_2$'] #'DAD $\ell_2$',

colors = {'real':'black','LINOCS':'blue', '1 step':'orange', 'Observed':'gray',
          'DAD': 'red',
     'DAD $\ell_2$': 'darkorange',
    'DAD reweigh': 'tomato',
    'reweigh $\ell_2$': 'coral'}

lw = 4
title_size = 30
predict_order_name = 'Prediction Order'
train_order_name = 'Train Order'
label_params = {'fontsize': title_size}
titles = ['Real', 'LINOCS', '1 step']
########################################################
# Multi-Step Finder
########################################################

def identify_system_from_data(max_apply, max_predict ):
    # max apply = integrate into the model optimization
    # max_predict - how long to predict?

    pass


