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

# 1-step vs. LINOCS
![image](https://github.com/user-attachments/assets/64aa9a0d-5a3b-45dd-a8d7-60f6d156bdc1)

# MSE
![image](https://github.com/user-attachments/assets/d829a151-2e85-4715-a318-0375e81f712a)

# Test different training and prediction orders
![image](https://github.com/user-attachments/assets/8fc46929-e0f5-45ed-af10-640844024019)
