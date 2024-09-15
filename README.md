# LINOCS
Code for  Mudrik, N., Yezerets, E., Chen, Y., Rozell, C., &amp; Charles, A. (2024). Linocs: Lookahead inference of networked operators for continuous stability. arXiv preprint arXiv:2404.18267.


### General main functions:

##### Main Functions (for LINOCS and plotting):  
`_main_functions_lookahead_linocs.py_`

#### Basic functions (unrelated basic functions to use):  
`_basic_function_lookahead.py_`

![image](https://github.com/NogaMudrik/LINOCS/blob/main/illustration_figure_LINOCS.png?raw=true)
## How to run?

# EXAMPLE 1: LINEAR EXPERIMENT WITH OFFSET
![image](https://github.com/user-attachments/assets/13ec0811-4550-4761-b3d1-adc0ccc57839)
- Please use the [google colab example notebook](https://colab.research.google.com/drive/1Ef30kC-68DGDsHQ4yMPGaFLyazC7lQrZ?usp=sharing) for an example and further explanations.
- Specific example for training Linear-LINOCS is below:
```
K_b = 20
max_ord = 250

func_w = np.exp
sigma_w = 0.01
w = [] 
sigma_w_b = 1
w_b = [] 

opt_A, opt_b,b_hats = train_linear_system_opt_A(x_noisy, max_ord, Bs_main = [], constraint = [], w_offset = True, cal_offset=True, weights=w, infer_b_way='each', K_b = K_b, w_b = w_b
                              )
offset_hat_opt = np.mean(opt_b, 1)

"""
full lookahead prediction
"""
x_look_LINOCS = propagate_dyn_based_on_operator(x0,opt_A, offset = offset_hat_opt, max_t = T)

"""
1-step prediction
"""
x_step_LINOCS = one_step_prediction(x_noisy, np.dstack([opt_A]*T), offset = offset_hat_opt)
```

### 1-step vs. LINOCS
![image](https://github.com/user-attachments/assets/64aa9a0d-5a3b-45dd-a8d7-60f6d156bdc1)

### MSE
![image](https://github.com/user-attachments/assets/d829a151-2e85-4715-a318-0375e81f712a)

### Test different training and prediction orders
![image](https://github.com/user-attachments/assets/8fc46929-e0f5-45ed-af10-640844024019)



# Example 2 - dLDS-LINOCS: 
## See demo/tutorial in notebook `https://github.com/NogaMudrik/LINOCS/blob/main/run_dLDS_Lorenz_example.ipynb` 
### importantly, this requires installing pylops version 1.18.2:
#### `!pip install pylops==1.18.2`

To train dLDS:
```
np.random.seed(seed)
M = np.random.randint(3, 10)    
K = np.random.randint(1, 5)    
l1 = np.random.rand()*0.2 #1.2 # 0.9# 2 #  #5
decor_w = np.random.rand()*0.1
max_iters = 70
save_path = os.getcwd()
start_l1 = np.random.randint(0, 5)
start_l1  = np.random.choice([start_l1 , 0])

data_run = lorenz_mat


    
l2_F = np.random.rand()
freq_update_F = 5
cs_h, F_h, F_full_h, cs_full_h = train_dlds_LINOCS(data_run, F0 = [], M = M, max_iters = max_iters, l1 = l1, 
                                                   to_save = True, save_path = save_path, 
                                                    seed = seed, start_l1 = start_l1, K = K,freq_save = 1,
                                                   l1_int = 1,freq_update_F = freq_update_F,
                                                    interval_round = 1, l2_F = l2_F,  with_identity =  with_identity, decor = decor, addi_save = 'example', decor_w = decor_w 
                                    )
```
