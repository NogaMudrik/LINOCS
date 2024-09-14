from dlds_LINOCS import *

all_trials = False 

data = 'synth'
if data == 'bat':
    bat_data = np.load(r'data_bat_xmin_0_xmax_n_ymin_0_ymax_n.npy')
    if not all_trials:
        data_run = bat_data[:,:,12]
    else:
        data_run = bat_data
elif data == 'IBL':

    data_run = np.load(r'IBL_rates.npy')
    if not all_trials:
        data_run = data_run[:5,:12]
    else:
        data_run = data_run
     
elif 'synth':
    pass
else:
    raise ValueError('data unkonwn!')    



seed = int(str(datetime2.now()).split('.')[-1])

home_or_nersc = 'h' if 'MATERIALS' in os.getcwd() else 'n'  
save_path = os.getcwd() 
 
with_identity = False 
decor = False 

if 'synth' not in data:
    normalize_mean_0 = False 
    if not normalize_mean_0:
        norm_z = np.random.choice([False,True])
    else:
        norm_z = False    
    decor_w = np.random.rand()    
else:
    normalize_mean_0 = False 
    norm_z = False    
    
addi_save = 'ide_'  + str(with_identity) + '_decor_' + str(decor) + '_norm0_' + str(normalize_mean_0) + '_normZ_' + str(norm_z)
today = str(datetime2.today()).split()[0]    


norm_z = False 
normalize_mean_0 = False 
random_params = False
if random_params:
    with_identity = False 
else:
    with_identity = False 
addi = today + '_new'
fix_c = False 
fix_F = False 

if fix_c:
    init_c = True
else:
    init_c = False 

if fix_F:
    init_F = True
else:
    init_F = False 

 
save_path = save_path +  os.sep +  'dLDS_' + data + os.sep + addi + os.sep + 'seed_%d'%seed
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
"""
create data
"""
if home_or_nersc == 'h':
    cur_pah = os.getcwd()
    os.chdir(r'E:\ALL_PHD_MATERIALS\CODES\LOOKAHEAD_DYNAMICS\switching_systems')
    from  create_synthetic_data_to_slds import *
    os.chdir(cur_path)
else:
    from  create_synthetic_data_to_slds import *
    
theta = np.pi*0.1
T = 990
period_min = 270 
period_max = 380
   
if with_identity:
    F = [create_rotation_mat(theta = theta, axes = axis, dims = 3) - np.eye(3) for axis in ['x','y','z']]
else:
    F = [create_rotation_mat(theta = theta, axes = axis, dims = 3) for axis in ['x','y','z']]
reps = np.random.randint(period_min, period_max, 9)
reps_sum = np.cumsum(reps)
cs = np.zeros((3, T))
last = 0
for c_rep, rep in enumerate(reps_sum):
    print(rep)
    c_rep_i = np.mod(c_rep, 3)
    if rep <= cs.shape[1]:
        
        cs[c_rep_i,last:rep] = 1
        last = rep
    elif last < cs.shape[1]:
        cs[c_rep_i, last:T] = 1
        last = rep
    else:
        break
cs = gaussian_convolve(cs,18, direction = 1)
cs = cs / np.sum(cs, 0)     
cc_gauss_d = cs
F_3d = np.dstack([np.sum(np.dstack([cs[i, t]*F[i] for i in range(cs.shape[0])]),2) for t in range(T)] ) #np.repeat(np.dstack(F), reps, axis = 2)
x0 = np.random.rand(3)# data_run[:,0]
if with_identity:
    F_new = F.copy()

    data_run = create_reco_new(x0, cs, F_new, type_reco='lookahead' , with_identity=with_identity)
    
else:
    data_run = create_reco_new(x0, cs, F, type_reco='lookahead'  , with_identity=with_identity)
    #propagate_dyn_based_on_operator(np.random.rand(3), F_3d )
   

if fix_c:
    c_init = cc_gauss_d
else:
    c_init = []
    
if fix_F:
    F_init = F 
    M = len(F)
else:
    F_init = []
    np.random.seed(seed)
    M = np.random.randint(3, 5)  

  

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
random_params = False 
if with_identity:
    F_norm = 0.44
else:
    F_norm = 1
if random_params:
    
    K = 0
    additional_update = np.random.choice([True, False])
    
    include_opt = np.random.choice([True, False])

    start_l1 = np.random.randint(0, 5)
    start_l1  = np.random.choice([start_l1 , 0])
    l1_init = start_l1
    max_iters = np.random.randint(100, 900)   
    l2_F = 0 #np.random.rand()*0.2
    freq_update_F = np.random.randint(1,5)
    plot_mid = True # False
    if home_or_nersc == 'h':
        plot_freq = 1   
    else:            
        plot_freq = 1#5
    style_treat_k = 'single'
    change_params_dict = {'l2': np.random.choice([0,0.9+0.09*np.random.rand()]), 'l1':0.99+0.009*np.random.rand(), 'l_smooth':1+0.09*np.random.rand(),
                          'l_smooth_time':1+0.09*np.random.rand(), 'std_noisy_c':0.9+0.09*np.random.rand()}
    change_params_min_vals = {'l2':0, 'l1':1, 'l_smooth_iters':0.03,'l_smooth_time':0.03, 'std_noisy_c':0.001}
    change_params_max_vals  = {'l2':8, 'l1':2.5, 'l_smooth_iters':5,'l_smooth_time':5, 'std_noisy_c':0.1}
    l2 = 0.5*np.random.rand()# 0.2
    l1 = 1+0.2*np.random.rand()#1.9
    thres_er = 0.001
    l_smooth_time = np.random.rand()*2 
    l_smooth_iters = np.random.rand()*0.5
    with_m_avg = np.random.choice([True, False])
    avg_wind = np.random.randint(3,50) 
    force_er_dec =  np.random.choice([True, False])
    std_noisy_c =  np.random.rand()*0.1 
    decor =  np.random.choice([True, False])
    max_interval_k = np.random.randint(3,10) 
    decor_w = np.random.rand()*0.1
    to_norm_F =  np.random.choice([True, False]) 
    with_hard_thres = True
    error_thres_hard_thres = 1000
    freq_hard_thres = 1 
    weights_style  =np.random.choice(['dec', 'eq'])
    metric_to_move = np.random.choice(['corr', 'mse'])
else:
    plot_freq = 1
    metric_to_move = 'corr'
    error_thres_hard_thres = 1000 
    freq_hard_thres = 1 
    with_hard_thres = False
    K = 50    

    additional_update = True
    
    include_opt = False
    

    start_l1 = np.random.randint(0, 5)
    start_l1  = np.random.choice([start_l1 , 0])
    l1_init = start_l1
    max_iters = 200
    l2_F = 0 
    freq_update_F = 5
    plot_mid =True 
    style_treat_k = 'single'
    change_params_dict = {'l2':0.99, 'l1':0.9999, 'l_smooth':1.01,'l_smooth_time':1.1, 'std_noisy_c':0.9999}
    change_params_min_vals = {'l2':0, 'l1':1, 'l_smooth_iters':0.03,'l_smooth_time':0.03, 'std_noisy_c':0.001}
    change_params_max_vals  = {'l2':8, 'l1':2.5, 'l_smooth_iters':5,'l_smooth_time':5, 'std_noisy_c':0.1}
    l2 =0
    l1 = 1.5
    

    thres_er = 0.001
    l_smooth_time = 0.1
    l_smooth_iters = 0
    with_m_avg = True 
    avg_wind = 5
    force_er_dec =  False 
    std_noisy_c = 0.05 
    decor =False
    max_interval_k = 1
    decor_w = 0.7
    to_norm_F = True
    weights_style  = 'dec'


params_dict = {
'with_identity':with_identity,
'K': K,
'l1': l1,
'additional_update': additional_update,
'include_opt': include_opt,
'start_l1': start_l1,
'l1_init': l1_init,
'max_iters': max_iters,
'l2_F': l2_F,
'freq_update_F': freq_update_F,
'plot_mid': plot_mid,
'plot_freq': plot_freq,
'style_treat_k': style_treat_k,
'change_params_dict': change_params_dict,
'change_params_min_vals': change_params_min_vals,
'change_params_max_vals': change_params_max_vals,
'l2': l2,
'thres_er': thres_er,
'l_smooth_time': l_smooth_time,
'l_smooth_iters': l_smooth_iters,
'with_m_avg': with_m_avg,
'avg_wind': avg_wind,
'force_er_dec': force_er_dec,
'std_noisy_c': std_noisy_c,
'decor': decor,
'max_interval_k': max_interval_k,
'decor_w': decor_w,
'to_norm_F': to_norm_F,
'weights_style': weights_style,
'x0':x0,
'data_run':data_run,
'cc_gauss_d':cc_gauss_d,
'F':F,
'cs':cs,
'M':M,
'F_3d':F_3d
}

np.save(save_path + os.sep + 'params.npy',    params_dict )




cs_h, F_h, F_full_h, cs_full_h , errors_steps, errors_full, k_change= train_dlds_LINOCS(data_run, F0 = F_init, c0 = c_init, M = M, max_iters = max_iters, l1 = l1, 
                                                                                        to_save = True, save_path = save_path, 
                                                   seed = seed, start_l1 = start_l1, K = K,freq_save = 1,
                                                   l1_int = 1,
                                                   freq_update_F = freq_update_F,
                                                   interval_round = 1, l2_F = l2_F,  
                                                   with_identity =  with_identity, decor = decor, addi_save = addi_save, decor_w = decor_w  , plot_mid = plot_mid, 
                                                   fix_F = fix_F, fix_c = fix_c, F_ground_truth = F, include_opt = include_opt, to_norm_F = to_norm_F,style_treat_k=style_treat_k,
                                                   l_smooth_time =  l_smooth_time ,  l_smooth_iters  =  l_smooth_iters, change_params_dict = change_params_dict, c_ground_truth = cc_gauss_d, thres_er = thres_er,
                                                   with_m_avg= with_m_avg, avg_wind = avg_wind, l2 = l2, 
                                                   force_er_dec = force_er_dec, std_noisy_c = std_noisy_c, additional_update = additional_update,
                                                   max_interval_k = max_interval_k, change_params_max_vals =change_params_max_vals, 
                                                   change_params_min_vals = change_params_min_vals, 
                                                   plot_freq = plot_freq,
                                                   with_hard_thres = with_hard_thres,
                                                   error_thres_hard_thres = error_thres_hard_thres,
                                                   freq_hard_thres =  freq_hard_thres, F_norm = F_norm, metric_to_move = metric_to_move
                                                   
                                    )


dict_save = {'cs_h': cs_h, 'F_h': F_h, 'F_full_h' : F_full_h, 'cs_full_h': cs_full_h, 'l1':l1, 'M':M, 'max_iters':max_iters, 'seed':seed, 'start_l1':start_l1, 'K':K, 
             ' l2_F': l2_F, 'decor' : decor, 'addi_save ': addi_save, 'with_identity':with_identity, ' decor_w': decor_w}
np.save(save_path + os.sep + 'dict_save%s__%d_%d.npy'%(addi_save , seed, M), dict_save)



