# -*- coding: utf-8 -*-

from basic_function_lookahead import *
import os

# Get the full path of the currently executing script
script_path = os.path.abspath(__file__)

# Print the full path of the script
print("Full path of the script:", script_path)
T = 2000
fac_mul = 0.2
c1 = np.sin(fac_mul*0.01*np.arange(T)) + np.cos(fac_mul*0.02*np.arange(T))
c2 = np.sin(fac_mul*0.002*np.arange(T)) + np.sin(fac_mul*0.01*np.arange(T))
c3 =  np.cos(fac_mul*0.002/2*np.arange(T))
cs = np.vstack([c1,c2,c3])
#cs = cs/np.sum(cs**2, 1).reshape((-1,1))
cs_gauss = gaussian_convolve(cs,direction = 1)
#plt.plot(cs_gauss.T)
cs_thres = np.hstack([keep_thres_only(cs[:,t],1).reshape((-1,1)) for t in range(T)])

#plt.plot(cs_thres.T)
cs_thres = np.hstack([keep_thres_only(cs[:,t],2).reshape((-1,1)) for t in range(T)])
#plt.plot(cs_thres.T)
cs_gauss = gaussian_convolve(cs_thres,direction = 1)
cs_thres = np.hstack([keep_thres_only(cs_gauss[:,t],2).reshape((-1,1)) for t in range(T)])
#plt.plot(cs_thres.T)
cs_gauss = gaussian_convolve(cs_thres,direction = 1)
#plt.plot(cs_gauss.T)
cs_gauss = gaussian_convolve(cs_thres,direction = 1, wind = 14)
#cs_gauss[0,:int(T/2)] = -cs_gauss[0,:int(T/2)] 
#cs_gauss[2,int(T/2):] = -cs_gauss[2,int(T/2):] 
cs = 0.996*cs_gauss / np.sqrt(np.sum(np.abs(cs_gauss)**2,0)).reshape((1,-1))#(fac*np.sqrt(np.sum(np.abs(cs_gauss)**1.8, 0)     ))
cs = gaussian_convolve(cs,27, direction = 1)
plt.figure()
plt.plot(cs.T)
plt.legend([1,2,3])


theta = 0.1
#F = [create_rotation_mat(theta = theta, axes = axis, dims = 3) - np.eye(3) for axis in ['x','y','z']]
F = [create_rotation_mat(theta = theta, axes = axis, dims = 3) for axis in ['x','y','z']]
#cs[0,:] = 1
#cs[1:,:] = 0
F_3d = np.dstack([np.sum(np.dstack([cs[i, t]*F[i] for i in range(cs.shape[0])]),2) for t in range(T)] ) #np.repeat(np.dstack(F), reps, axis = 2)
np.random.seed(0)
x0 = np.random.rand(3)*10# data_run[:,0]
if with_identity:
    F_new = F.copy()
    #F_new.append(np.eye(data_run.shape[0]))
    #cs_new = np.vstack([cs, np.ones((1, cs.shape[1]))])
    data_run = create_reco_new(x0, cs, F_new, type_reco='lookahead' , with_identity=with_identity)
    #propagate_dyn_based_on_operator(np.random.rand(3), F_3d + np.expand_dims( np.eye(3),2))
else:
    data_run = create_reco_new(x0, cs, F, type_reco='lookahead'  , with_identity=with_identity)
    #propagate_dyn_based_on_operator(np.random.rand(3), F_3d )

plt.figure();plt.plot(data_run.T)
np.save(r'E:\ALL_PHD_MATERIALS\CODES\LOOKAHEAD_DYNAMICS\november 2023' + os.sep + 'coeffs_complex.npy', cs)
#np.save('coeffs_complex.npy', cs_gauss)