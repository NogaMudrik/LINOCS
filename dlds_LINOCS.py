# -*- coding: utf-8 -*-
"""

dlds functions
"""


from main_functions_lookahead_linocs import *

from numpy import linalg as LA



def find_c_new_march(x, F, l1, l2, l_smooth, params = {'threshkind':'soft','solver':'spgl1','num_iters':100}, with_identity = False):
    cs_new = []
    
        
    for t in range(x.shape[1]-1):

        right =  np.hstack([(F_i @ x[:,t].reshape((-1,1))).reshape((-1,1)) for F_i in F]) # right
        if with_identity:
            left = x[:,t+1].reshape((-1,1)) - x[:,t].reshape((-1,1))
        else:
            left = x[:,t+1].reshape((-1,1))
        
        dim = right.shape[1]
        if l2> 0:
            
            right = np.vstack([right  , np.eye(dim)])
            left = np.vstack([left, np.zeros((dim,1))])
            
            
        if l_smooth and t > 0:
            right = np.vstack([right  , np.eye(dim)])
            left = np.vstack([left, cs_cur.reshape((-1,1))])
            
        cs_cur =   solve_Lasso_style(right, left, l1, params = params)

        cs_new.append(cs_cur.reshape((-1,1)))
    cs = np.hstack(cs_new)
    return cs
    
def find_c_new_march_lookahead(x, F, l1, l2, l_smooth, K = 0, weights = [], params = {'threshkind':'soft','solver':'spgl1','num_iters':100}, with_identity = False):
    cs_new = []
    weighs = np.ones(K+1)
        
    for t in range(x.shape[1]-1):
        if K >0:

            pass
        K_cur = np.min([t, K])
        if K_cur == 0:
            right =  np.hstack([(F_i @ x[:,t].reshape((-1,1))).reshape((-1,1)) for F_i in F]) # right
        elif K_cur > 0:
            right = np.zeros([0, len(F)]) 

            for k_i in range(K_cur+1): 
                
                x_next =  x[:,t-k_i]
                for next_step in range(k_i):
                    x_next  = np.sum(np.dstack([cs_new[t-k_i+next_step][i]*F[i] for i in range(cs_new[0].shape[0])]),2) @ x_next.reshape((-1,1)) # herehere
                right_new = np.hstack([(F_i @ x_next.reshape((-1,1))).reshape((-1,1)) for F_i in F])


                right = np.vstack([right, right_new])

        else:
            raise ValueError('how?')
            
            
        if with_identity:
            left = x[:,t+1].reshape((-1,1)) - x[:,t].reshape((-1,1))
        else:
            left = x[:,t+1].reshape((-1,1))
        if K_cur > 0:
            left = np.vstack([left]*(K_cur+1))

        dim = right.shape[1]
        if l2> 0:
            
            right = np.vstack([right  , np.eye(dim)])
            left = np.vstack([left, np.zeros((dim,1))])
            
            
        if l_smooth and t > 0:
            right = np.vstack([right  , np.eye(dim)])
            left = np.vstack([left, cs_cur.reshape((-1,1))])

        cs_cur =   solve_Lasso_style(right, left, l1, params = params)

        cs_new.append(cs_cur.reshape((-1,1)))
    cs = np.hstack(cs_new)
    return cs    


def train_dlds_LINOCS(x, M = 0, F0 = [], c0  = [], params_lasso = {} ,seed = 5, thres_er = 0.1, K = 5, max_iters = 10, 
                      l_smooth_int = 1.01, l1_int = 0.99, l_smooth = 0.1,
                      l1 = 1.2, with_m_avg = True, interval_round = 1, 
                      style_treat_k = 'single', to_save = False, save_path = '.', style_compare = 'step', freq_save = 2, freq_update_F = 5,
                      avg_wind = 5, start_l1 = 0, l2_F = 0.2, with_identity = True, decor = False, decor_w = 0.1, addi_save = '' , F_frob = False, plot_mid = False,
                      F_ground_truth = [],
                      fix_c = False, fix_F = False, include_opt = True, to_norm_F = False, l2 = 0, change_params_dict = {},
                      c_ground_truth = [], std_noisy_c = 0.1,
                      c_inference_style = 'each_x',  init_F_style = 'random', force_er_dec = True,  l_smooth_time = 1, 
                      l_smooth_iters  = 0, additional_update = False, thres_next_K = 0.1, max_interval_k = 2, condition_stop = 'small_full_er',  weights_style ='dec',                      
                      change_params_max_vals = {}, 
                     change_params_min_vals = {}, plot_freq = 5    ,         with_hard_thres =  False,
                             error_thres_hard_thres = 0.3, max_active_cs = 2,
                             freq_hard_thres = 4   , F_norm = 1  ,         non_dec = True,
                             additional_update_last = False, metric_to_move = 'corr' 
                      ):
    """  
    x can be either 1 observation (in this case 2d mat) or k observations (in this case 3d mat). 
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    F0 : TYPE, optional
        DESCRIPTION. The default is [].
    c0 : TYPE, optional
        DESCRIPTION. The default is [].
    params_lasso : TYPE, optional
        DESCRIPTION. The default is {}.
    seed : TYPE, optional
        DESCRIPTION. The default is 0.
    condition_stop can be 'k' or 'small_full_er'
    Returns
    -------
    None.

    """
    params_lasso = {**{'threshkind':'soft','solver':'spgl1','num_iters':10}, **params_lasso}
    l_smooth = l_smooth_time

    if len(x.shape) == 3:
        x_3d = x.copy()
        multi_trial = True
    else:
        multi_trial = False
        

        
    if M == 0 and checkEmptyList(F0):
        raise ValueError('you must provide M or F0')
    elif (M != 0) and (M != len(F0)) and not  checkEmptyList(F0):
        raise ValueError('M should be the same len as F0')
    elif not  checkEmptyList(F0):
        M = len(F0)
        
    dim = x.shape[0]
    """
    initialize F
    """
    if checkEmptyList(F0):
        if init_F_style =='butches':
            edges = np.linspace(0, x.shape[1]-1, M+1).astype(int)
            if not multi_trial:
                F = [x[:,edges[i]:edges[i+1]] @ np.linalg.pinv(x[:,edges[i]+1: edges[i+1]+1])  for i in range(M)]
            else:
                F = [x[:,edges[i]:edges[i+1],0] @ np.linalg.pinv(x[:,edges[i]+1: edges[i+1]+1,0])  for i in range(M)] 
            
        elif init_F_style == 'random':
            F = []
            for m in range(M):
                np.random.seed(m)
                F.append(np.random.rand(dim, dim))
        if to_norm_F:
            F = [f_i*F_norm/ LA.norm(f_i, 2) for f_i in F]

            
    else:
        F = F0.copy() 
    
    
    

    
    
    """
    while loop to train c and F
    """
    counter = 0
    er = 100
    T = x.shape[1]
    
    if multi_trial:
        dim3 = x_3d.shape[2]
        cs_full = {dd: [] for dd in range(dim3)}
        F_full = [] 
        cs_3d = {dd:[]  for dd in range(dim3)} 
        K_each = {dd:K  for dd in range(dim3)} 
        trial_was = []
        
    else:        
        cs_full = [] 
        #if not fix_F:
        F_full = []
        if not fix_c:
            cs = []
        else:
            cs = c0
    x_K = np.array([np.nan])
    
    additional_const = True

    K_or = K
    K_cur = 0
    errors_steps = {}
    errors_full = {}
    k_changes = {}
    
    
    if condition_stop == 'small_full_er':        
        cond_stop = er > thres_er 
    elif condition_stop == 'k':
        cond_stop = K_cur < K
        
        
    while (cond_stop and counter < max_iters and (np.isnan(x_K).sum() < 8)) or additional_const :
        

        if not multi_trial:
            additional_const = False
        else:
            additional_const = counter < dim3
        if multi_trial:
            
            trial_take = np.mod(counter, dim3)
            x = x_3d[:,:, trial_take]
            cs = cs_3d[trial_take]
            K = K_each[trial_take]
            if trial_take not in trial_was:
                trial_was.append(trial_take)

        if counter < start_l1:
            l1_ef = 0 
        elif counter == start_l1:
            l1_ef = l1_int*l1
        else:
            l1_ef *= l1_int

        """
        update c
        """

        if not fix_c:
          
        
            
            cs = find_c_new_march_lookahead(x, F, l1, l2, l_smooth,  K = K_cur, params = params_lasso, with_identity=with_identity)
            
            if   with_hard_thres and np.mod(counter, freq_hard_thres) == 0 and er < error_thres_hard_thres :

                    cs = np.hstack([keep_thres_only(cs[:,t], cs.shape[0] - max_active_cs, direction = 'lower', perc = False, num = True).reshape((-1,1)) for t in range(cs.shape[1])])

                    """
                    reestimate
                    """
                    c_new  = []
                    for t in range(x.shape[1]-1):
                        Fx = np.hstack([(F_i @ x[:,t].reshape((-1,1))).reshape((-1,1)) for F_i in F]) 
                        c_cols = np.where(cs[:,t] != 0)[0]
                        Fx_take = Fx[:, c_cols]
                        c_in_cols = solve_Lasso_style(Fx_take, x[:,t+1], l1, params = params_lasso) 
                        c_cur = cs[:,t]
                        c_put = c_cur.copy()
                        c_put[c_cols] = c_in_cols
                        c_new.append(c_put.reshape((-1,1)))
                    cs = np.hstack(c_new)
                    
                              
                    
                    
            if plot_mid and np.mod(counter, plot_freq) == 0:
                fig = plt.figure()
                plt.plot(cs.T)
                plt.title(str(l1) + 'here here directly after update l1' + str(l1))
                plt.show()
                
    
                fig.savefig(save_path + os.sep + 'cbefore_%d.png'%counter)
                plt.close()

        
            if with_m_avg:
                cs = mov_avg(cs, wind = avg_wind, axis = 1)

            if plot_mid and np.mod(counter, plot_freq) == 0:
                if checkEmptyList(c_ground_truth): #                   "
                    fig, ax = plt.subplots(figsize = (20,5))
                    ax.plot(cs.T)
                    fig.savefig(save_path + os.sep + 'c_%d l1 %.2f.png'%(counter,l1))
                    fig.suptitle('after avg if avg avg %s l2 %.2f'%(str(with_m_avg), l1))
                    plt.close()
                else:
                    fig, ax = plt.subplots(1,2,sharey = True,figsize = (20,5))
                
                    ax[0].plot(cs.T)
                    ax[1].plot(c_ground_truth.T)
                    fig.suptitle('after avg if avg avg %s l2 %.2f'%(str(with_m_avg), l1))
                    fig.savefig(save_path + os.sep + 'c_%d l1 %.2f.png'%(counter,l1))
                    
                    plt.close()
                    
                    
        if not fix_F:
            if np.mod(counter, freq_update_F) == 0 or not multi_trial or len(x.shape) < 3:

                F, cs = opt_F(x, cs, F, seed = seed + K, style_compare = style_compare, l2_F = l2_F, with_identity=with_identity, decor = decor,
                              decor_w = decor_w, F_frob = F_frob, fix_c = fix_c, include_opt = include_opt , to_norm_F = to_norm_F, std_noisy_c = std_noisy_c,
                              force_er_dec = force_er_dec , F_norm = F_norm
                              )
                if to_norm_F:
                    F = [f_i*F_norm / LA.norm(f_i, 2) for f_i in F]

                    if not fix_c and additional_update:
                        cs = find_c_new_march(x, F, l1, l2, l_smooth,  params = params_lasso, with_identity=with_identity)
                        
            
            F_full.append(F)

        if multi_trial:    
            K_each[trial_take] = np.min([K+interval_round, x.shape[1]])  
            cs_3d[trial_take] = cs
            cs_full[trial_take].append(cs)
            
        else:
            
            cs_full.append(cs)
        if not multi_trial:    
            x_K_step = x.copy()    
          
            cur_plus_int = K_cur + max_interval_k
            errors_full[counter]  = {}
            for k_plot in range(np.min([cur_plus_int, K])+1):    
                x_K_step = create_reco_new(x_K_step, cs, F, type_reco='step' , with_identity=with_identity)

                if k_plot == 0:
                    x_K_step_first = x_K_step.copy()
                er_k = np.nanmean((x_K_step - x)**2)
                errors_full[counter][k_plot] = er_k
                
                update_k = False#break
                if (np.mean((x_K_step - x)**2) < thres_next_K and metric_to_move == 'mse' ) or (metric_to_move == 'corr' and spec_corr(x_K_step.flatten(), x.flatten()) > 0.9):
                    k_plot_works = k_plot
                    update_k = True
                   

            if update_k:

                choose_K_list= [K_or, K_cur+interval_round, x.shape[1], k_plot_works]
                if non_dec:
                    K_cur = np.max([K_cur,np.min(choose_K_list) ])
                else:
                    
                    K_cur = np.min(choose_K_list) 
            
        if not fix_c and additional_update_last:
            """
            update based on only the last one
            """
            cs_new = []
            for t in range(x.shape[1]-1):
                cs_cur =  find_c_t(x_K_step, F,t, cs, 0 , l1 = 0 , l2 = l2 ,
                                      l_smooth_time =  l_smooth_time ,  l_smooth_iters  =  
                                      0 if counter == 0 else l_smooth_iters,   c_inference_style = 'each_x', 
                                      weights = [], with_identity = with_identity, c_t_minus_1 = cs_cur if t > 0 else [], 
                                      weights_style  =  weights_style  )[0].reshape((-1,1)) 
                cs_new.append(cs_cur)
            cs = np.hstack(cs_new)
            
            
            
        
        k_changes[counter] = K_cur

            
        x_K = create_reco_new(x[:,0].reshape((-1,1)), cs, F, with_identity = with_identity)     

        if np.nanmax(np.abs(x_K)) < np.max(np.abs(x))*500:
            er = np.nanmean((x_K - x)**2)
        else:
            er = 10**5

            cs += np.random.rand(*cs.shape)*std_noisy_c*2
        corr_cur = spec_corr(x, x_K)
        errors_full[counter] = er
        if to_save:
            print('saving!! DICT: (in %s)'%save_path)
            
            
            if np.mod(counter, freq_save) == 0:
                
                                
                if multi_trial:
                    np.save(save_path + os.sep + 'results_iter_%d_corr_%.2f_er_%.2f.npy'%(counter, corr_cur, er), 
                            {'F': F, 'cs':cs_3d, 'F_full':F_full, 'cs_full':cs_full, 'counter':counter, 'er': er, 'l1':l1_ef, 'decor_w':  decor_w,
                             'decor':decor, 'with_identity':with_identity,
                             'errors_steps' : errors_steps,
                             'errors_full' : errors_full,
                             'k_changes' : k_changes, 'K':K,'K_cur':K_cur,
                             'l2_F':l2_F, 'x_0':x[:,0]})
                    
                    
                else:
                    np.save(save_path + os.sep+ 'results_iter_%d_corr_%.2f_er_%.2f.npy'%(counter, corr_cur, er), 
                            {'F': F, 'cs':cs, 'F_full':F_full, 'cs_full':cs_full, 'counter':counter, 'er': er, 'l1':l1_ef, 'decor_w':  decor_w, 'decor':decor, 
                             'with_identity':with_identity, 'l2_F':l2_F, 
                             'errors_steps' : errors_steps,
                             'errors_full' : errors_full, 'K':K,'K_cur':K_cur,
                             'k_changes' : k_changes, 'x':x, 'x_K':x_K                             
                             })
                    

                

                
            print('saving!! (in %s)'%save_path)
            if plot_mid and np.mod(counter, plot_freq) == 0:
                fig, ax = create_3d_ax(1,4, params = {'figsize':(20,4)})
                plot_3d(x, ax = ax[0])      
                plot_3d(x_K, ax = ax[1])     
                ax[1].set_title('k cur %d'%K_cur)
                
                plot_3d(x, ax = ax[2], params_plot = {'color':'gray'})      
                plot_3d(x_K_step, ax = ax[2])  
                
                plot_3d(x, ax = ax[3], params_plot = {'color':'gray'})      
                plot_3d(x_K_step_first, ax = ax[3])  
                fig.tight_layout()
                fig.savefig(save_path + os.sep + 'x_%d.png'%counter)
                        
                plt.close()
                
            if plot_mid and np.mod(counter, plot_freq) == 0: 
                fig, ax = plt.subplots(1,4, figsize = (20,4))
                sns.heatmap(x, ax = ax[0], center = 0,vmin = -0.6, vmax = 0.6 )      
                sns.heatmap(x_K, ax = ax[1], vmin = -0.6, vmax = 0.6, center = 0)     
                ax[1].set_title('k cur %d'%K_cur)
                sns.heatmap(x_K_step, ax = ax[2], vmin = -0.6, vmax = 0.6, center = 0)          
                sns.heatmap(x_K_step_first, ax = ax[3], vmin = -0.6, vmax = 0.6, center = 0)      
                
                fig.savefig(save_path + os.sep + 'x_heat_%d.png'%counter)
                        
                plt.close()
            if plot_mid and np.mod(counter, plot_freq) == 0:

                if  checkEmptyList(F_ground_truth):
                    fig, ax = plt.subplots(1, len(F), figsize = (5*len(F),5)); 
                else:
                    fig, axs = plt.subplots(2, len(F), figsize = (5*len(F),10)); 
                    ax = axs[0]
                [sns.heatmap(F_i, square = True, annot = False, ax = ax[i], center = 0) for i, F_i in enumerate(F)]
                
                if not  checkEmptyList(F_ground_truth):
                    [sns.heatmap(F_i, square = True, annot = False, ax = axs[1,i], center = 0) for i, F_i in enumerate(F_ground_truth)]
                
                fig.savefig(save_path + os.sep + 'F_%d.png'%counter)
                plt.close()
        

        counter += 1
        """
        change_params
        """
                
        # Usage example
        l2 = update_parameter(l2, change_params_dict, change_params_max_vals, change_params_min_vals)
        l_smooth_time = update_parameter(l_smooth_time, change_params_dict, change_params_max_vals, change_params_min_vals)
        l1 = update_parameter(l1, change_params_dict, change_params_max_vals, change_params_min_vals)
        l_smooth_iters = update_parameter(l_smooth_iters, change_params_dict, change_params_max_vals, change_params_min_vals)
        std_noisy_c = update_parameter(std_noisy_c, change_params_dict, change_params_max_vals, change_params_min_vals)
            
            
            
    

            
        if condition_stop == 'small_full_er':        
            cond_stop = er > thres_er 
        elif condition_stop == 'k':
            cond_stop = K_cur < K
            
            

    if not multi_trial:
        return cs, F, F_full, cs_full, errors_steps, errors_full, k_changes
    else:
        return cs_3d, F, F_full, cs_full, errors_steps, errors_full, k_changes
         
    
    
    


def update_parameter(parameter, change_params_dict, change_params_max_vals, change_params_min_vals, default_value=np.inf):
    new_param = parameter * change_params_dict.get(parameter, 1)
    
    if new_param > change_params_max_vals.get(parameter, default_value):
        new_param = change_params_max_vals.get(parameter, parameter)
    elif new_param < change_params_min_vals.get(parameter, 0):
        new_param = change_params_min_vals.get(parameter, parameter)
    
    return new_param

  
    
def opt_function_F(F_hat, x_plus, xcs, num_zeros, with_identity = False, x_minus = [], decor = False, decor_w = 0.1, F_frob = False):
    
    F_hat = np.reshape(F_hat, (x_plus.shape[0], xcs.shape[0]))

    if not with_identity:
        addi1 = np.sum(( x_plus - F_hat @ xcs)**2)
    else:
        if checkEmptyList(x_minus):
            raise ValueError('you must provide x_minus if with_identity')
            
        addi1 = np.sum(( x_plus - np.eye(len(x_plus.flatten())) @  x_minus  - F_hat @ xcs)**2)
    if decor:
       
        addi1 = addi1 +  decor_w*np.sum(lists2list([[spec_corr(f1, f2) if f1_c != f2_c else 0 for f1_c, f1 in enumerate(F)] for f2_c, f2 in enumerate(F)]))
    if F_frob:
        addi1 =  addi1 + 0.2*addi1*np.sum(np.abs(F_hat))
    return  addi1
    
def opt_F(x,c, F, num_zeros = 12, std_noisy_c = 0.1, seed = 0, style_compare = 'lookahead', l2_F = 0, with_identity = False,
          decor = False, decor_w = 0.1, F_frob = False, fix_c = False, include_opt = False, 
          to_norm_F = False, force_er_dec = True, style_opt ='LS', F_norm = 1):

    dim = x.shape[0]
    if force_er_dec:
        counter_max = 3
    else:
        counter_max = 1
    if checkEmptyList(F) and decor:
        raise ValueError('cannot decor if preivious F not provided')
    M = c.shape[0]
    x_plus = x[:,1:]
    x_minus = x[:,:-1]
    if len(F) > 0:
        reco = create_reco_new(x, c, F, style_compare, with_identity  = with_identity) #'step')
        er_former = np.mean((reco - x)**2)
    else:
        er_former = np.inf
        
    er_new = np.inf
    
    counter = 0
    
    if decor:
        F_former = F.copy() 
        F_formers = []
        for shift in range(1, M): 
            F_former = F_former[:-1] + [F_former[0] ]
            F_former_inv = [np.linalg.pinv(f_i) for f_i in F_former]
            F_formers.append(np.hstack(F_former_inv))
        left_formers_h = np.hstack( F_formers)
        right_formers_h = np.hstack([np.eye(dim*M)]*(M-1))
        
        
    while er_new >= er_former and counter < counter_max:
        
        xcs = np.hstack([(np.tile(x[:,i].reshape((-1,1)), (c.shape[0],1)) * np.repeat(c[:,i].reshape((-1,1)), x.shape[0], axis = 0)).reshape((-1,1))
               for i in range(x.shape[1]-1)])


        if style_opt == 'LS' or not decor or decor_w == 0:
            """
            ls based update
            """
            if np.mod(seed, 5)  == 4 and include_opt:
               
                F = np.hstack([F_i for F_i in F])
                F_h = F + np.random.rand(*F.shape)*0.1
                F_h = F_h.flatten()

                F_hat = minimize(opt_function_F, F_h, method = 'BFGS', args = (x_plus, xcs, num_zeros,  with_identity, x_minus, decor, decor_w,  F_frob)).x

                F_hat = np.reshape(F_hat, (x_plus.shape[0], xcs.shape[0]))
            else:
                if l2_F == 0:
                    if not with_identity:
                        left =  x_plus
                        right =   xcs
                        
                        if decor:
                            left = np.hstack([left, decor_w*left_formers_h ])
                            right = np.hstack([right, decor_w *right_formers_h ])
                        F_hat = left @ np.linalg.pinv(right)

                    else:
                     
                        left = (x_plus - np.eye(len(x_plus)) @ x_minus)
                        right = xcs
                        if decor:
                            left = np.hstack([left, decor_w*left_formers_h ])
                            right = np.hstack([right, decor_w*right_formers_h ])
                        F_hat = left   @ np.linalg.pinv(right)
                    
                else:
                    raise ValueError('why?~!?')
                    if  with_identity:
                        F_hat = np.hstack([x_plus - np.eye(len(x_plus)) @ x_minus , np.zeros((x_plus.shape[0], xcs.shape[0]))]) @ np.linalg.pinv(np.hstack([xcs,  l2_F*np.eye(xcs.shape[0])]))
                    else:
                        F_hat = np.hstack([x_plus, np.zeros((x_plus.shape[0], xcs.shape[0]))]) @ np.linalg.pinv(np.hstack([xcs,  l2_F*np.eye(xcs.shape[0])]))
                        
        elif style_opt == 'gradient':
            """
            Frobenious norm
            
            """
            Bis = []
            Bis_dim = len(F) * F[0].shape[0]
            p = F[0].shape[0]
            for counter_f, f in enumerate(F):
                for counter_row in range(len(F)):
                    if counter_row != counter_f:
                        next_B = np.zeros((Bis_dim, p))
                        print(counter_row*p, counter_row*(p+1))
                        next_B[counter_row*p:counter_row*p + p,:] = f
                        Bis.append(next_B)
                
                
            sum_bi = np.hstack(Bis).sum(1).T
            
            psi = xcs.copy()
            if is_1d(psi):                
                psi =  psi.reshape((-1,1))
                
            
            x_next = x_plus.copy()
            if is_1d(x_next):    
                x_next = x_next.reshape((-1,1))
            sum_fiedlity = 2*x_next @ psi.T
            psi_psi_T = psi @ psi.T
            if l2_F == 0: 
                F_hat = 0.5*(sum_fiedlity + sum_bi*decor_w) @ psi_psi_T
                
            else:
                F_hat = 0.5*(sum_fiedlity + sum_bi*decor_w) @ (psi_psi_T + np.eye(Bis_dim))
                
                
                
        else:
            raise ValueError('invalid style optimization')
            
        edges = np.linspace(0,F_hat.shape[1], M+1).astype(int)
        F = [F_hat[:, edge:edges[i+1]] for i, edge in enumerate(edges[:-1])]
        
        reco = create_reco_new(x, c, F, style_compare, with_identity = with_identity)
        er_new = np.mean((reco - x)**2)
        counter += 1
        np.random.seed(counter*seed)
        if er_new >= er_former and not fix_c:
            c += np.random.rand(*c.shape)*std_noisy_c

        
        
        
    return F,c



def find_c_full(x, F, K = 4, c_prev = [], l1 = 0.2, l2 = 0, l_smooth = 0, c_inference_style = 'opt', weights = [], seed = 0, num_rounds = 5, interval_round = 1,
                weights_style = 'dec', style_treat_k = 'single', with_identity = False):
    np.random.seed(seed)
    
    if checkEmptyList(weights):
        if weights_style == 'dec':
            weights = (np.arange(K) + 1)[::-1].astype(float)
        elif weights_style == 'eq':
            weights =  np.ones(K)       
        else:
            raise ValueError('weights style undefined')
        

        
    c_avgs = {}
    estimates_c_last = {}
    estimates_full = {}
    if style_treat_k == 'multi':
        iterator = np.arange(1, K + 1, interval_round)
    else:
        iterator = [K]
    for k_i in iterator:

        estimates_full[k_i] = []
        c_avgs[k_i] = []
        estimates_c_last[k_i] = {k:[] for k in range(K)}
        for t in range(x.shape[1] - 1):

            c_avg,estimates_c_last_t, estimates_x_former  = find_c_t(x, F, t, c_prev, np.min([t+1,k_i]), l1, l2, l_smooth,
                                  c_inference_style, weights, with_identity = with_identity,  weights_style  =  weights_style )

            estimates_c_last[k_i] = estimates_c_last_t
            c_avgs[k_i].append(c_avg.reshape((-1,1)))
           
            if checkEmptyList(c_prev):
                
                c_prev = c_avg.reshape((-1,1))
            elif c_prev.shape[1] > t:
                
                c_prev[:,t] = c_avg.flatten()
            else:
                c_prev = np.hstack([c_prev, c_avg.reshape((-1,1))])

        c_avgs[k_i] = np.hstack(c_avgs[k_i])
    return c_avgs, estimates_c_last
        


def find_c_t(x, F,t, c_prev, K = 4, l1 = 1.1, l2 = 0,l_smooth_time = 0, l_smooth_iters = 0, c_inference_style = 'each_x',
             weights = [], with_identity = False, c_t_minus_1 = [],  weights_style = 'eq'):

    print(l_smooth_iters)
    if l_smooth_iters > 0 and checkEmptyList(c_prev):
        raise ValueError('you must provide c_prev is l_smooth_iters')
    if l_smooth_time > 0 and checkEmptyList(c_t_minus_1 ) and t > 0:
        raise ValueError('you must provide c_t_minus_1  is l_smooth_iters')
    if t == 0:
        l_smooth_time = 0
    if c_inference_style != 'each_c' and  c_inference_style != 'opt' and  c_inference_style != 'each_x':
        raise ValueError( 'c_inference_style not defined')
    if checkEmptyList(weights):
        if weights_style == 'dec':
            weights = (np.arange(K+ 1)+1)[::-1].astype(float)
        elif weights_style == 'eq':
            weights = np.ones(K+1)
        else:
            raise ValueError('unrecognized weights_style')
    weights = weights[:np.min([t+1, K+1])]    
    if t >= x.shape[1] - 1:
        raise ValueError('t is too large!')
    if checkEmptyList(c_prev):
        print('no lookahead!')

    if t > 0 and K > 0:
        K_take = np.min([K, t])
        As = find_local_operators(F, zs  = [], cs = c_prev[:, t - K_take  : t])
    
    estimates_x_former  = []
    estimates_c_last  = []
    
    itera = np.arange(np.max([1,np.min([t+1, K])]))
    

    
    for k in itera: 
        if k == 0 or t == 0:
            x_former = x[:,t]
        else:    
            if K_take != As.shape[2]:
                print('k %d'%K_take)
                print('As.shape[2] %d'%As.shape[2])
                raise ValueError('something incorrect')

            x_former = k_step_prediction(x[:,t - k].reshape((-1,1)), As[:,:,-k:], k, store_mid = False, t = -1, offset = [])[:,-1]    

        estimates_x_former.append(x_former.reshape((-1,1)))

        if  c_inference_style == 'each_c' or k == 0:

            Fx =  np.hstack([(F_i @ x_former.reshape((-1,1))).reshape((-1,1)) for F_i in F])
            if l2 == 0:
                left = Fx;             right = x[:,t+1].reshape((-1,1)) 
            else:
                left = np.vstack([Fx, l2*np.eye(Fx.shape[1])])
                right = np.vstack([x[:,t+1].reshape((-1,1)), l2*np.zeros((Fx.shape[1], 1))])
            """
            smooth_time
            """

            if l_smooth_time and t > 0:
               
                left = np.vstack([left, l_smooth_time*np.eye(left.shape[1]) ])
                right = np.vstack([right, l_smooth_time*c_t_minus_1.reshape((-1,1)) ])
                

                
            
            """
            smooth iters
            """

            if l_smooth_iters:

                left = np.vstack([left, l_smooth_iters*np.eye(Fx.shape[1])])
                right = np.vstack([right, l_smooth_iters*c_prev[:,t].reshape((-1,1)) ])       
                
            if l1 == 0:
                c_last_hat = np.linalg.pinv(left) @ right
            else:
                c_last_hat = solve_Lasso_style(left, right, l1)
            estimates_c_last.append(c_last_hat)
            if k == 0:
                former_est = c_last_hat.copy()
                
    weights /= np.sum(weights)           
    if  c_inference_style == 'each_x' :

        x_next = np.vstack([x[:,t+1].reshape((-1,1))]*len(estimates_x_former))
        dim = x.shape[0]

        x_next = x_next*np.repeat(weights[:k+1], dim).reshape((-1,1))
        
        Fx =  np.vstack([np.hstack([weights[count_est]*F_i @ x_j.reshape((-1,1)) for F_i in F]) 
                           for count_est, x_j in  enumerate(estimates_x_former )                      
                         ])
        
        
        if with_identity:
            identities =  np.vstack([weights[count_est]*np.eye(len(x_j.flatten())) @ x_j.reshape((-1,1)) 
                               for count_est, x_j in   enumerate(estimates_x_former )                      
                             ])

        else:
            print('?!?!?!')
            

        if l2 == 0:
            left = Fx;        
            if with_identity:
                
                x_right = x_next.reshape((-1,1)) - identities.reshape((-1,1))
            else:
                x_right = x_next.reshape((-1,1))
            right = x_right
                
        else:
            left = np.vstack([Fx, l2*np.eye(Fx.shape[1])])
            
            if with_identity:
                x_right = x_next.reshape((-1,1)) - identities.reshape((-1,1))
            else:
                x_right =  x_next.reshape((-1,1))
            
            right = np.vstack([x_right, l2*np.zeros((Fx.shape[1], 1)).reshape((-1,1))])

        if t > 0 and l_smooth_time:
            left = np.vstack([left, l_smooth_time*np.eye(Fx.shape[1])])

            right = np.vstack([right, l_smooth_time*c_t_minus_1.reshape((-1,1)) ])       
            
            
        """
        ITERS SMOOTH
        """
        if l_smooth_iters == 0: 
            pass
        else:
            left = np.vstack([left, l_smooth_iters*np.eye(Fx.shape[1])])

            right = np.vstack([right, l_smooth_iters*c_prev[:,t-1].reshape((-1,1)) ])       
            
        if l1 == 0:

            c_last_hat = np.linalg.pinv(left) @ right
        else:

            c_last_hat = solve_Lasso_style(left, right, l1, params = {'threshkind':'soft','solver':'spgl1','num_iters':100})
        estimates_c_last.append(c_last_hat)
        if k == 0:
            former_est = c_last_hat.copy()
        c_avg = c_last_hat   

            
    if k != 0:    
        if c_inference_style == 'opt':
            c_avg = minimize(c_optimize, former_est, method = 'BFGS',
                             args = (x[:,t+1].reshape((-1,1)), F, estimates_x_former, c_prev[:,t-1] if t >= 1 else [], l2, l_smooth if t >= 1 else 0, l1, 
                                      weights)).x
            
            
        elif  c_inference_style == 'each_c' :
            c_avg = np.sum(np.hstack([weights[i]*estimates_c_last[i].reshape((-1,1)) for i in range(len(estimates_c_last))]),1)
            
            
    return c_avg, estimates_c_last, estimates_x_former
        


def solve_c_no_lookahead_each_x(x, F, c_prev = [], l2 = 0, l1 = 0, l_smooth = 0, t = -1, with_identity = False ):
    """
    Solve linear regression for each time step.
    
    Parameters:
    - x (numpy.ndarray): Matrix where each column represents a different time step.
    - F (list of numpy.ndarray): List of matrices.
    - c_prev (numpy.ndarray, optional): Previous solution. Default is an empty list.
    - l2 (float, optional): L2 regularization parameter. Default is 0.
    - l1 (float, optional): L1 regularization parameter. Default is 0.
    - l_smooth (float, optional): Smoothness regularization parameter. Default is 0.
    - t (int, optional): Time step. Default is -1.
    
    Returns:
    - numpy.ndarray: Matrix of solutions for each time step.
    """
    if t == -1:
       iterator  = np.arange(x.shape[1]-1) 
    else:
       iterator  =  [t]  
    c_num = len(F)    
    cs = []
    for t in iterator:
        x_j = x[:,t]
        x_next = x[:,t+1]
        Fx = np.hstack([F_i @ x_j.reshape((-1,1)) for F_i in F])
        
        if with_identity:
            x_next = x_next.reshape((-1,1)) - (np.eye(len(x_next)) @ x_j).reshape((-1,1))
            
            
            
        if l2 == 0:
            left = Fx;             right = x_next.reshape((-1,1))
        else:
            left = np.vstack([Fx, l2*np.eye(Fx.shape[1])])
            right = np.vstack([x_next.reshape((-1,1)), l2*np.zeros((c_num, 1)).reshape((-1,1))]).reshape((-1,1))
        
        if l_smooth == 0 or t == 0:
            pass #left = left;             right = right 
        else:
            c_prev = cs[-1]
            left = np.vstack([left, l_smooth*np.eye(Fx.shape[1])])

            right = np.vstack([right, l_smooth*c_prev.reshape((-1,1)) ])       
            
        if l1 == 0:
            c_last_hat = np.linalg.pinv(left) @ right
        else:
            c_last_hat = solve_Lasso_style(left, right, l1)
        cs.append(c_last_hat.reshape((-1,1)))    
    cs = np.hstack(cs)
    return cs
        
        



def c_optimize(c,  x_plus, F, x_estimates, c_form = [], l2 = 0, l_smooth = 0, l1 = 0, weights = []):

    if checkEmptyList(weights):
        weights = np.ones(len(x_estimates))
        weights /= np.sum(weights)
    K = len(x_estimates)

    Fxs = [np.hstack([(F[i]@ x_estimate.reshape((-1,1))).reshape((-1,1)) for i in range(len(F))])
           for k,x_estimate in enumerate(x_estimates)]

    addi_0 = np.sum([np.sum(weights[i]*(x_plus - Fx @ c)**2) for i,Fx in enumerate(Fxs)])
    
    
    if l_smooth > 0 and checkEmptyList(c_form):
        raise ValueError('if l_smooth you must provide c')
        
    if l_smooth > 0:
        addi_1 = l_smooth*((c_form - c)**2).sum()
    else:
        addi_1 = 0
    
    if l1 > 0:
        addi_2 = l1*np.abs(c).sum()
    else:
        addi_2 = 0        
        
    if l2 > 0:
        addi_3 = l2*(c**2).sum()
    else:
        addi_3 = 0
    return addi_0 + addi_1 + addi_2 + addi_3
        
        
    
    



to_run = False
if to_run:
    cc_dlds2, estimates_c_last2 = find_c_full(x_noisy, F, l1 = 1,  K = 2, c_inference_style='each', c_prev = [], l2 = 0, l_smooth = 0.5)    