# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:30:56 2023

@author: noga mudrik

NOTES ON PREDICTIONS:
    -  ONE STEP:
        one_step_prediction(x, As, t = -1, k = -1, t_start = -1, t_end = -1, offset = []):  
    - K STEP:
        k_step_prediction_linear(x_noisy, opt_A, order_predict, store_mid = True, t = -1, offset = opt_b) 
    - FULL LOOKAHEAD:
        propagate_dyn_based_on_operator(x_noisy, opt_A, offset = opt_b, max_t = x_noisy.shape[0] - 1 )
        
"""
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.ticker as ticker
import itertools
import numpy.linalg as linalg
from sklearn import linear_model
from itertools import product  
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.optimize import linear_sum_assignment
import time
import os
#import cmath
from sklearn.linear_model import OrthogonalMatchingPursuit
in_local = True
"""
make latex
"""
# from matplotlib import rc
# import matplotlib.pylab as plt

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

try:
    import pylops
except:
    print('did not load pylops')
def find_c_from_operators(operators, F):
    F_mat = np.hstack([F_i.flatten().reshape((-1,1)) for F_i in F])
    F_mat_inv = np.linalg.pinv(F_mat)
    operators_stack = np.hstack([operators[:,:,t].flatten().reshape((-1,1)) for t in range(operators.shape[2])])
    return F_mat_inv @ operators_stack


def create_sparse_lambda(len_vec, i):
    vec = np.zeros(len_vec)
    vec[i] = 1
    return vec
    


def lists2list(xss)    :
    return [x for xs in xss for x in xs] 

def find_all_roots_including_complex(n,k):
    # find x s.t. x^n = k
    abs_k = np.abs(k)
    n_k_basis = abs_k**(1/n)
    theta = cmath.phase(k)
    cos_part = [np.cos((theta+2*n_i*np.pi)/n) for n_i in range(n)]
    sin_part = [np.sin((theta+2*n_i*np.pi)/n) for n_i in range(n)]
    sol = [n_k_basis*(cos_part_i + 1j*sin_part_i) for cos_part_i, sin_part_i in zip(cos_part, sin_part)]
    return sol

def eigen_dec(w,v, num_remove = 0):
    # v is a mat cpaturing the evecs in its cols
    if not isinstance(w, (list, tuple, np.ndarray)):
        w = [w]
    if isinstance(w, tuple)    :
        w = list(w)
    smallest = np.argsort([np.abs(w_i) for w_i in w])
    if num_remove > 0:
        v[:,smallest[:num_remove]] = 0
    res =  v @ np.diag(w) @ np.linalg.pinv(v)

    return res
    


    
  
def fractional_matrix_power_all_sols(B, n, return_all_optns = False):
    """
    Find all possible matrices A such that A^n = B.

    Parameters:
    - B (numpy.ndarray): The target matrix B.
    - n (int): The power to which matrices A should be raised to obtain B.

    Returns:
    list: A list of matrices A satisfying A^n = B.

    Notes:
    - The function utilizes the eigenvalue decomposition of B to find possible solutions.
    - The eigenvalue decomposition is performed using numpy's eig function.
    - For each eigenvalue, all possible roots, including complex roots, are calculated using find_all_roots_including_complex.
    - The function then generates all combinations of these roots for all eigenvalues.
    - Finally, eigen_decs is called to construct eigen decompositions for each combination.

    See Also:
    - find_all_roots_including_complex: Function to find all roots, including complex roots, for a given eigenvalue.
    - eigen_dec: Function to construct the eigen decomposition for a given set of eigenvalues and eigenvectors.
    """
    w, v = np.linalg.eig(B)

    #for every eval find all options
    sols_for_each_w = [find_all_roots_including_complex(n,w_i) for w_i in w]
    
    # all_optns is a list of tuples of n**dim_dynamics. e.g. if B is 3X3 and n = 4, # of solutions will be 4**3
    all_optns = list(product(*sols_for_each_w))
   
    eigen_decs = [eigen_dec(w_opt,v)  for w_opt in all_optns]
    if return_all_optns:
        return eigen_decs, all_optns
    return eigen_decs


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
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],                            
                                [-np.sin(theta),np.cos(theta)]])
        else:
            raise ValueError('axes is invalid')
        
    else: 
        raise ValueError('dims should be 2 or 3')
    return rot_mat    
    
    
    
def pad_mat(mat, pad_val = np.nan, size_each = 1, axis = 1):
    # size each can be an integer or a list
    if isinstance(size_each,(list, tuple, np.ndarray)):
        size_left = size_each[0]
        size_right = size_each[1]
    else:
        size_left = size_each
        size_right = size_each        
        
    if axis == 1:
        pad_left = np.ones((mat.shape[0], size_left))*pad_val
        pad_right = np.ones((mat.shape[0], size_right))*pad_val
        mat = np.hstack([pad_left, mat, pad_right])
        
    elif axis == 0:
        pad_left = np.ones((size_left, mat.shape[1]))*pad_val
        pad_right = np.ones((size_right, mat.shape[1]))*pad_val
        mat = np.vstack([pad_left, mat, pad_right])  
        
    elif axis == 2:
        #each_pad = np.ones((mat.shape[0], mat.shape[1], size_each))*pad_val
        pad_left = np.ones((mat.shape[0], mat.shape[1], size_left))*pad_val
        pad_right = np.ones((mat.shape[0], mat.shape[1], size_right))*pad_val
        mat = np.dstack([pad_left, mat, pad_right])        
    else:
        raise ValueError('undefined axis for padding')
    return mat

def mov_avg(c, axis = 1, wind = 5):
    #print('As.shape')
    #print(c.shape)
    if len(c.shape) == 2 and axis == 1:
        if np.mod(wind,2) == 0:
            wind += 1
        wind_p = int((wind - 1)/2)
        c_shape = c.shape[1]
        c = pad_mat(c, pad_val = np.nan, size_each = wind_p, axis = 1)
        return np.hstack([np.nanmean( c[:,i:i+wind],1).reshape((-1,1))
              for i in range(c_shape)])
    elif len(c.shape) == 2 and axis == 0:
        return mov_avg(c.T, axis = 1).T
    
    elif len(c.shape) == 3 and axis == 2: # and axis == 0:. thos os pver to,e
        print('is hhere!!')
        c_shape = c.shape[2]
        wind_p = int((wind - 1)/2)
        c_shape = c.shape[1]
        c = pad_mat(c, pad_val = np.nan, size_each = wind_p, axis = 2)
        return np.dstack([np.nanmean(c[:,:,t:t + wind],  axis) for t in range(c.shape[2])  ])
    else:
        raise ValueError('how did you arrive here? data dim is %s, axis %d'%(str(c.shape), axis))
        
        

def checkEmptyList(obj):
    """
    Check if the given object is an empty list.

    Args:
        obj (object): Object to be checked.

    Returns:
        bool: True if the object is an empty list, False otherwise.

    """    
    return isinstance(obj, list) and len(obj) == 0

def d3tod32(mat)        :
    mat_2d = np.vstack([
     
     np.vstack([
       mat[i,j,:]
      
      for j in range(mat.shape[1])
      ]) for i in range(mat.shape[0])
     
     ] )
    return mat_2d
    

def remove_edges(ax, include_ticks = False, top = False, right = False, bottom = True, left = True):
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], 
               ylabel_params = {'fontsize':19},zlabel_params = {'fontsize':19}, xlabel_params = {'fontsize':19}, 
               title_params = {'fontsize':29}, format_xticks = 0, format_yticks = 0):
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
  if len(legend)       > 0:  ax.legend(legend)
  if format_xticks > 0:
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_xticks))
  if format_yticks > 0:      
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_yticks))
      


def create_lorenz(psi0 = [0.1, 0.1, 0], dt = 0.010, max_t = 3,
                  sigma = 10, beta = 8/3, rho = 28, return_operators = True, option = 1):
    # {'psi0': [0.2127086755529508, 2.6381260266588527, 1.0635433777647538],
    # 'dt': 0.024647819173282542,
    # 'max_t': 6,
    # 'sigma': 10.425417351105901,
    # 'beta': 2.546042008886284,
    # 'rho': 26.97792145774131,
    # 'return_operators': True,
    # 'option': 2}
    psi = np.array(psi0).reshape((-1,1))
    """
    define A
    """
    As = []
    for t in np.arange(0, max_t, dt):
        A = create_lorenz_mat(psi[:,-1], sigma, beta, rho, option = option)
        psi_next =  (A*dt + np.eye(A.shape[0])) @ psi[:,-1]
        psi = np.hstack([psi, psi_next.reshape((-1,1))])
        if return_operators:
            As.append(A*dt + np.eye(A.shape[0])) 
    if return_operators:
        return psi, As
    return psi
    
def create_3d_ax(num_rows, num_cols, figsize = (), params = {}):
    if 'figsize' not in params and len(figsize) > 0:
        params['figsize'] = figsize
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax    


def remove_background(ax, grid = False, axis_off = True):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not grid:
        ax.grid(grid)
    if axis_off:
        ax.set_axis_off()
def plot_2d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot'):
    # 
    if checkEmptyList(ax):
        fig, ax = plt.subplots(1,1, **params_fig)
    if type_plot == 'plot':    
        ax.plot(mat[0], mat[1], **params_plot)
    else:
        ax.scatter(mat[0], mat[1], **params_plot)
def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot', to_return = False):
    # 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':    
        scatter = ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        scatter = ax.scatter(mat[0], mat[1], mat[2], **params_plot)
    if to_return:
        return scatter
    
def create_colors(len_colors, perm = [0,1,2], style = 'random', cmap  = 'viridis', seed = 0, reduce_green = 0.4):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    np.random.seed(seed)
    if style == 'random':
        colors = np.random.rand(3, len_colors)
        colors[1] = colors[1]*reduce_green
    else:
        cmap = plt.get_cmap(cmap) 
        # Create an array of values ranging from 0 to 1 to represent positions in the colormap
        positions = np.linspace(0, 1, len_colors)

        colors = [cmap(pos) for pos in positions]

    return colors
    

    
    
def create_lorenz_mat(psi, sigma, beta, rho, option = 1):
    """
    Generate the matrix A for the Lorenz system based on the given parameters.

    Parameters:
    - psi (list): A list containing the initial values [x0, y0, z0].
    - sigma (float): Parameter controlling the rate of change of x.
    - beta (float): Parameter controlling the behavior of the system.
    - rho (float): Parameter controlling the convective flow.

    Returns:
    numpy.ndarray: The matrix A for the Lorenz system.

    Example:
    >>> psi = [1, 2, 3]
    >>> sigma = 10
    >>> beta = 8/3
    >>> rho = 28
    >>> result = create_lorenz_mat(psi, sigma, beta, rho)
    >>> print(result)
    array([[-10,  10,   0],
           [ 28,  -1,  -1],
           [  0,   3,  -8/3]])
    """    
    x = psi[0]
    if option == 1:
        row1 = [-sigma, sigma, 0]
        row2 = [rho, -1, -x]
        row3 = [0, x, -beta]
    else:
        z = psi[2]
        row1 = [-sigma, sigma, 0]
        row2 = [rho - z, -1, 0]
        row3 = [0, x, - beta]
        
    A = np.vstack([ row1, row2, row3])
    return A
    
    
def add_basic_axes(ax = [], fig = [], max_z = 1, max_x = 1, max_y = 1, 
                   min_z = 0, min_x = 0, min_y = 0,  
                   params_subplot = {},
                   params_plot = {'color' : 'black', 'w': 4,  'ls':'-' ,'mutation':20, 
                                  'arrowstyle':"-|>", 
                                  'linewidth': 2}, remove_back = True, 
                   remove_grid = True,  remove_axes = False, remove_ticks = False):    
    # 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1, 1, params_subplot)
    
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z 
    # """
    # x
    # """ 
    # ax.arrow3D(min_x,min_y,min_z,
    #       dx,dy,dz,
    #        mutation_scale=params_plot['mutation'],
    #        arrowstyle=params_plot['arrowstyle'],
    #        linestyle=params_plot['ls'], color = params_plot['color'], 
    #        linewidth = params_plot['linewidth'])        
    # #y 
    # ax.arrow3D(min_x,min_y,min_z,      dx,dy,dz,
    #         mutation_scale=params_plot['mutation'],
    #         arrowstyle=params_plot['arrowstyle'],
    #         linestyle=params_plot['ls'], color = params_plot['color'],
    #         linewidth = params_plot['linewidth'])   
    
    # #z 
    # ax.arrow3D(min_x,min_y,min_z,
    #      dx,dy,dz,
    #         mutation_scale=params_plot['mutation'],
    #         arrowstyle=params_plot['arrowstyle'],
    #         linestyle=params_plot['ls'], color = params_plot['color'],
    #         linewidth = params_plot['linewidth'])
    
    to_remove_back(ax, remove_back, remove_grid,  remove_axes, remove_ticks)
        
        
def to_remove_back(ax, remove_back = True, remove_grid = True, remove_axes = False, remove_ticks = False):
    if remove_back:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    if remove_grid:
        ax.grid = False
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if remove_axes:
        ax.axis('off')
    

def find_c_from_operators(operators, F):
    F_mat = np.hstack([F_i.flatten().reshape((-1,1)) for F_i in F])
    F_mat_inv = np.linalg.pinv(F_mat)
    operators_stack = np.hstack([operators[:,:,t].flatten().reshape((-1,1)) for t in range(operators.shape[2])])
    return F_mat_inv @ operators_stack

    
#def     


"""
3d plotting
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)    


def is_1d(mat):
    if isinstance(mat,list): mat = np.array(mat)
    elif isinstance(mat, np.ndarray): pass
    else: raise ValueError('Mat must be numpy array or a list')
    return np.max(mat.shape) == len(mat.flatten())

def save_fig(name_fig,fig, save_path = '', formats = ['png','svg'], params_save = {'transparent':True}) :
    if len(save_path) == 0:
        save_path = os.getcwd()
        
    [fig.savefig(save_path + os.sep + '%s.%s'%(name_fig, format_i), **params_save) for format_i in formats]
        
        

def solve_Lasso_style(A, b, l1, params = {}, lasso_params = {},random_state = 0, nouter = 50,
                      ):
  """
      Solves the l1-regularized least squares problem
          minimize (1/2)*norm( A * x - b )^2 + l1 * norm( x, 1 ) 
          
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    l1 : float
        scalar between 0 to 1, describe the reg. term on the cofficients.    
    params : TYPE, optional
        DESCRIPTION. The default is {}.
    lasso_params : TYPE, optional
        DESCRIPTION. The default is {}.
    random_state : int, optional
        random state for reproducability. The default is 0.

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    x : np.ndarray
        the solution for min (1/2)*norm( A * x - b )^2 + l1 * norm( x, 1 ) .

  lasso_options:
               - 'inv' (least squares)
               - 'lasso' (sklearn lasso)
               - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
               - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
               - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)       
               - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
               - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)
               
               
               - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
  """ 
  #print(params['solver'])
  params = {**{'threshkind':'soft','solver':'spgl1','num_iters':10}, **params}
  print(params['solver'])
  #print( params['solver'].lower() )
  if np.isnan(A).any():
      print('there is a nan in A')
      #input('ok? solve_Lasso_style')
  if len(b.flatten()) == np.max(b.shape):
      b = b.reshape((-1,1))
  if 'solver' not in params.keys():
      warnings.warn('Pay Attention: Using Default (inv) solver for updating A. If you want to use lasso please change the solver key in params to lasso or another option from "solve_Lasso_style"')
  params = {**{'threshkind':'soft','solver':'spgl1','num_iters':10}, **params}

  if params['solver'] == 'inv' or l1 == 0:

      #input('jgkljglkfjkgjf')
      if is_1d(A):
          pinv_A = linalg.pinv(A).reshape((-1,1))

      else:
          pinv_A = linalg.pinv(A)
      x = pinv_A @ b.reshape((-1,1))

  elif params['solver'] == 'lasso' :
      #fixing try without warm start
    clf = linear_model.Lasso(alpha=l1,random_state=random_state, **lasso_params)

    #input('ok?')
    clf.fit(A,b.flatten() )     #reshape((-1,1))
    x = np.array(clf.coef_)

  elif params['solver'].lower() == 'fista' :
      Aop = pylops.MatrixMult(A)
  
      #if 'threshkind' not in params: params['threshkind'] ='soft'
      #other_params = {'':other_params[''],
      x = pylops.optimization.sparsity.FISTA(Aop, b.flatten(), niter=params['num_iters'],
                                             eps = l1 , threshkind =  params.get('threshkind') )[0]
  elif params['solver'].lower() == 'ista' :

      #fixing try without warm start
      if 'threshkind' not in params: params['threshkind'] ='soft'
      Aop = pylops.MatrixMult(A)
      x = pylops.optimization.sparsity.ISTA(Aop, b.flatten(), niter=params['num_iters'] , 
                                                 eps = l1,threshkind =  params.get('threshkind'))[0]
      
  elif params['solver'].lower() == 'omp' :
      #print(A.shape[1] - l1)
      #input('?')
      omp = OrthogonalMatchingPursuit(n_nonzero_coefs=A.shape[1] - l1, fit_intercept   = False)
      omp.fit(A,b)
      #Aop = pylops.MatrixMult(A)

      x  = omp.coef_ # pylops.optimization.sparsity.OMP(Aop, b.flatten(),                                                  niter_outer=params['num_iters'], sigma = l1)[0]     
  elif params['solver'].lower() == 'spgl1' :
      print('here spgl1!!!!!!!!!!')
      Aop = pylops.MatrixMult(A)
      x = pylops.optimization.sparsity.SPGL1(Aop, b.flatten(),iter_lim = params['num_iters'],  tau = l1)[0]      
      
  elif params['solver'].lower() == 'irls' :
   
      Aop = pylops.MatrixMult(A)
      
      #fixing try without warm start
      x = pylops.optimization.sparsity.IRLS(Aop, b.flatten(),  nouter = nouter, espI = l1)[0]      
  else:     
    raise NameError('Unknown update c type')  
  return x

import pandas as pd    

def count_freq(x, x_ind = 0, y_ind = 1, time_check = 5, return_rad = False):
    x_center = x.mean(1)[0]
    y_center = x.mean(1)[1]
    x1 = x[x_ind,5] -  x_center#[0]
    x2 = x[x_ind,6] - x_center#models_results_b['Observed'][0]
    y1 = x[y_ind,5] - y_center#models_results_b['Observed'][1]
    y2 = x[y_ind,6] - y_center# models_results_b['Observed'][1]    
    call_diff = np.tan(y2/x2) - np.tan(y1/x1) #/np.pi*180  #+ 180
    return call_diff if return_rad else call_diff/np.pi*180
    
def number_of_points_2_circles(num_samples, x, params_freq = {}, return_angle = False): 
    dW = count_freq(x, **params_freq)
    num_samples_per_circle = 360/dW   
    if not return_angle:
        return num_samples/num_samples_per_circle
    else:
        return num_samples/num_samples_per_circle,  dW
    
    
    
    
    
    
    
    
def pad_mat(mat, pad_val, size_each = 1, axis = 1):
    if axis == 1:
        each_pad = np.ones((mat.shape[0], size_each))*pad_val
        mat = np.hstack([each_pad, mat, each_pad])
    else:
        each_pad = np.ones((size_each, mat.shape[1]))*pad_val
        mat = np.vstack([each_pad, mat, each_pad])        
    return mat
    

def gaussian_array(length,sigma = 1 , to_norm_type = 'max' ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
        to_norm_type can be 'not', 'max', 'sum'
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    if to_norm_type == 'not':
        pass
    elif  to_norm_type == 'max':
       gaussian = gaussian / np.max(gaussian)
    elif  to_norm_type == 'sum':
        gaussian = gaussian / np.sum(gaussian)
    else:
        raise ValueError('?!')
        
    return gaussian


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

from scipy.sparse import coo_matrix  
def create_legend(dict_legend, size = 30, save_formats = ['.png','.svg'], 
                  save_addi = 'legend' , dict_legend_marker = {}, 
                  marker = '.', style = 'plot', s = 500, to_save = True, plot_params = {'lw':5},
                  save_path = os.getcwd(), params_leg = {}):
    fig, ax = plt.subplots()
    if style == 'plot':
        [ax.plot([],[], 
                 c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
    else:
        if len(dict_legend_marker) == 0:
            [ax.scatter([],[], s=s,c = dict_legend.get(area), label = area, marker = marker, **plot_params) for area in dict_legend]
        else:
            [ax.scatter([],[], s=s,c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
    ax.legend(prop = {'size':size},**params_leg)
    remove_edges(ax, left = False, bottom = False)
    fig.tight_layout()
    if to_save:
        [fig.savefig(save_path + os.sep + 'legend_areas_%s%s'%(save_addi,type_save)) 
         for type_save in save_formats]
        
        

      
def from_spike_times_to_rate(spike_dict, type_convert = 'discrete',
                             res = 0.01, max_min_val = [], return_T = False, 
                             T_max = np.inf, T_min = 0,  params_gauss = {}):
    """
    Converts spike times to firing rates.
    spike dict is dictionary of units vs spike times
    res is how much to mutiply it by
    Parameters:
    - spike_dict (dict): A dictionary of units vs spike times.
    - res (float): A value by which to multiply the spike times.
    - type_convert (str): Type of conversion to perform (default is 'discrete').
    - Ts (dict): Dictionary containing time indices.
    - Ns (dict): Dictionary containing neuron indices.
    - firings_rates_gauss (dict): Dictionary containing Gaussian-convolved firing rates.
    - firings_rates (dict): Dictionary containing firing rates.
    - max_min_val (list): List containing minimum and maximum values.
    - return_T (bool): Whether to return firing rate matrices (default is False).
    - T_max (float): Maximum time value (default is np.inf).
    - params_gauss (dict): Dictionary containing parameters for Gaussian convolution.
    
    Returns:
    - firing_rate_mat (ndarray): Matrix containing firing rates.
    - firing_rate_mat_gauss (ndarray): Matrix containing Gaussian-convolved firing rates.
    - return_T (bool): Whether to return firing rate matrices.
    
    import numpy as np


    """  
    if isinstance(spike_dict , (np.ndarray, list)):
        spike_dict = {1: spike_dict}       
        
        
    if T_min >= T_max:
        raise ValueError('T_min must be larger than T_max')
    if res != 1:
        spike_dict = {key:np.array(val) / res for key,val in spike_dict.items()}
    if T_min > 0:
        spike_dict = {key:val - T_min for key,val in spike_dict.items()}
        spike_dict = {key : val[val > 0] for key,val in spike_dict.items()}
        print(len(list(spike_dict.values())[0] ))
        #max_val = max_val - T_min 
        
    """
    make sure keys are continues
    """
    if set(np.arange(len(spike_dict))) != set(list(spike_dict.keys())):
        new_keys = np.arange(spike_dict)
        old_keys = list(spike_dict.keys())
        old2new = {old:new for old,new in zip(old_keys, new_keys)}
        spike_dict = {old2new[key]:val for key,val in spike_dict.items()}
    else:
        old2new = {}
    
    
    
    if checkEmptyList(max_min_val):
        min_val = np.min([np.min(val) for val in list(spike_dict.values())])
        max_val = np.max([np.max(val) for val in list(spike_dict.values())])
        #min_max_val = [min_val, max_val]
        
        
    N = len(spike_dict)
    # if (min_val < 0 and T_min == 0) or T_min > 0:
    #     if T_min == 0:
    #         T_min = min_val
    #     spike_dict = {key : val - T_min for key,val in spike_dict.items()}
    #     spike_dict = {key : val[val > 0] for key,val in spike_dict.items()}
        
    if T_min > 0:
        max_val = max_val - T_min     
    max_val = int(np.ceil(max_val))
    max_val = int(np.min([max_val, T_max]))
    firing_rate_mat = np.zeros((int(N) ,max_val))    

        
    if type_convert == 'discrete':         
        T_thres = T_max #- T_min
        tup_neurons_and_spikes = np.vstack([ np.hstack([np.array([neuron]*np.sum( times < T_thres )).reshape((-1,1)) , np.array(times[ times < T_thres]).reshape((-1,1)) ])
                                  for neuron, times  in spike_dict.items()])
        rows =  tup_neurons_and_spikes[:,0]
        cols =  tup_neurons_and_spikes[:,1]
        
        data = np.ones(len(rows))  # Assuming all values are 1
        sparse_mat = coo_matrix((data, (rows, cols)), shape=(N, max_val))
        
        # for count, (neuron, times) in enumerate(spike_dict.items()):
                
        #     times_within = times.astype(int) #- max_min_per_file[neural_key][0]

        #     max_ind = times_within[(times_within > T_min ) & (times_within < T_max)]

        #     firing_rate_mat[count, max_ind] += 1

        firing_rate_mat = sparse_mat.toarray()
        firing_rate_mat_gauss = gaussian_convolve(firing_rate_mat,  **params_gauss)
            
    if T_min > 0 :     
        firing_rate_mat = firing_rate_mat[:, T_min:]
        firing_rate_mat_gauss = firing_rate_mat_gauss[:, T_min:]
    if return_T:
        return  firing_rate_mat, firing_rate_mat_gauss, return_T
    return  firing_rate_mat, firing_rate_mat_gauss, old2new
            
        
        
        
        
        
def spec_corr(v1,v2, to_abs = True):
  """
  absolute value of correlation
  """
  corr = np.corrcoef(v1.flatten(),v2.flatten())
  if to_abs:
      return np.abs(corr[0,1])
  return corr[0,1]
    
def one_step_prediction(x, As, t = -1, k = -1, t_start = -1, t_end = -1, offset = []):  
    """
    here is 1 step (1_step) predicution for reconstructing the data
    """
    if checkEmptyList(offset):
        #print(len([1,2,3]) == 3)
        if is_1d(x) and  (len(As.shape) == 2 or As.shape[-1] == 1):
            if len(As.shape) == 3 and  As.shape[-1] == 1:
                As = As[:,:,-1]
            #print(As.shape)

            #print(len(As.shape))
            #print(len(As.shape) == 3)
            #print(As.shape[-1] == 1)
            return ( As @ x.reshape((-1,1)) ).reshape((-1,1)) # + offset.reshape((-1,1))   
            
            
        elif t == -1 :
            return np.hstack([x[:,0].reshape((-1,1))] + [ (As[:,:,t] @ x[:,t].reshape((-1,1))).reshape((-1,1)) 
                                                         for t in range(x.shape[1] - 1) ])
        elif t != -1 and k != -1:
            if t_start == -1:
                t_start = np.max([t - 1,0])         
            if t_end == -1:
                t_end = np.min([t + k - 1, As.shape[2]]) 
            return  np.hstack([ (As[:,:,t_i ] @ x[:,t_i ].reshape((-1,1))).reshape((-1,1)) 
                                                         for t_i in range(t_start, t_end + 1) ])
    
        
        else:
            raise ValueError('t and k should be both -1 or both not -1')
    else:
        if is_1d(offset):
            offset = offset.reshape((-1,1))
        if is_1d(x) and  (len(As) == 2 or As.shape[-1] == 1):
            if As.shape[-1] == 1:
                As = As[:,:,-1]
            return (x.reshape((-1,1)) @ As).reshape((-1,1)) + offset          
        else:

            if  len(As) == 2 or (len(As) == 3 and  As.shape[-1] == 1):
                T = x.shape[1] - 1
                As = np.dstack([As] * T)
            
                
            offset = np.hstack([offset]*As.shape[2])
            if t == -1 :
                return np.hstack([x[:,0].reshape((-1,1))] + [ (As[:,:,t] @ x[:,t].reshape((-1,1))).reshape((-1,1)) + offset[:,t].reshape((-1,1)) 
                                                             for t in range(x.shape[1] - 1) ])
    
            elif t != -1 and k != -1:
                if t_start == -1:
                    t_start = np.max([t - 1,0])         
                elif t_end == -1:
                    t_end = np.min([t + k - 1, As.shape[2]]) 
                return  np.hstack([ (As[:,:,t_i ] @ x[:,t_i ].reshape((-1,1)) + offset[:,t_i].reshape((-1,1))).reshape((-1,1)) 
                                                             for t_i in range(t_start, t_end + 1) ])
        
            
            else:
                raise ValueError('t and k should be both -1 or both not -1')
                
                
                
def d3tolist(F_3d):
    return [F_3d[:,:,i] for i in range(F_3d.shape[2])]
    



        
        
def k_step_prediction_linear(x, As, K, store_mid = True, t = -1, offset = []): 
    print('pay attention k_step does not store mid!')
    #print('jjjjjjjjjjjjjjjjjjjjjjjjj')
    # PAY ATTENTION T IS NOT INVOLVED HERE
    if K == 1 and checkEmptyList(offset):
        if len(As.shape) == 3 and As.shape[-1] == 1:
            As = As[:,:,0]
        if is_1d(x):
            return As @ x.reshape((-1,1))
        else:
            return As @ x
    if t != -1:
        raise ValueError('future implement!')
    x_partly = x[:,:-K]
    x0 = x[:,0].reshape((-1,1))
    #for k_i in range(K):
    if  checkEmptyList(offset):    
        x_k =  np.linalg.matrix_power(As,K) @ x_partly
    else:
        left1 = np.linalg.matrix_power(As,K)
        
        #print(As.shape)
        left2 = np.sum(np.dstack([
            np.linalg.matrix_power(As,k_i) for k_i in range(K)
            ]),2) @ offset.reshape((-1,1))
        # print(left2)
        # print('???????????????')
        # print(offset)
        # print('========================')
        left_full = np.hstack([left1, left2])
        right = np.vstack([x_partly , np.ones((1, x_partly.shape[1]))])
        x_k =  left_full @ right
    
    x_initial = [x0]
    x_former = x0
    """
    for these before K
    """
    for k_i in range(K-1):
         
        x_former = (As @ x_former).reshape((-1,1))
        if not  checkEmptyList(offset):     
            x_former = x_former + offset.reshape((-1,1))
        x_initial.append(x_former)
    x_initial = np.hstack(x_initial)
    return np.hstack([x_initial, x_k])



    

def k_step_prediction_depracated(x, As, K, store_mid = True, t = -1, offset = []):  
    if len(As.shape) == 2:
        return k_step_prediction_linear(x, As, K, store_mid , t , offset)
        #As = np.dstack([As]*x.shape[1])
        
    # IF t == -1: then it means that we need to consider the full duration.
    x = x.copy()
    if t == -1: # FOR THE FULL DURATION
        if store_mid:
            stores = []
        for k in range(K):
            x = one_step_prediction(x, As, offset = offset)    
            if store_mid:
                stores.append(x)
        if store_mid:
            return x, stores
        return x
    
    else: # CHECK ONLY EFFECT OF CHANGING As[:,:,t]
        if store_mid:
            stores = []
        for k in range(K):            
            t_start = np.max([t - 1,0])         
            t_end = np.min([t + k - 1, As.shape[2]])
            x_local = one_step_prediction(x, As, t , k, t_start , t_end , offset = offset)
            x[:,t_start : t_end+1] = x_local
           
            if store_mid:
                stores.append(x)
        if store_mid:
            return x, stores
        return x
    
    

    
    
def propagate_dyn_based_on_operator(x0, As, max_t = 0, offset = [], with_identity = False): # - MULTI STEP PREDICTION
    """
    Propagate the dynamic system based on a given set of operators for multi-step prediction.

    Parameters:
    - x0 (numpy.ndarray): Initial state vector.
    - As (numpy.ndarray): 2D or 3D array of operators. If 2D, it's broadcasted to create a 3D array for each time step.
      If 3D, the third dimension should match the number of time steps (max_t).
    - max_t (int): Maximum number of time steps for prediction.

    Returns:
    - numpy.ndarray: Array containing the propagated state vectors for each time step.

    Raises:
    - ValueError: If the third dimension of As does not match max_t.

    """
    if not with_identity:
        if max_t <= 0:
            if len(As.shape) == 2:
                raise ValueError('you must provide max_t if A constant')
            max_t = As.shape[2]
        if len(As.shape) == 2:
            As = np.dstack([As]*max_t)
        elif As.shape[2] != max_t:
            raise ValueError('Max t does not fit A')
            
        if checkEmptyList(offset):        
            x = x0.reshape((-1,1))
            for t in range(max_t):        
                x = np.hstack([x,  (As[:,:,t] @ x[:,-1].reshape((-1,1)) ).reshape((-1,1)) ])
            return x
        else:
            if is_1d(offset):
                offset = offset.reshape((-1,1))
                offset = np.hstack([offset]*max_t)
            if offset.shape[0] != As.shape[0]:
                raise ValueError('Offset shape does not match A shape?!')          
                
                
            if len(As.shape) == 2:
                As = np.dstack([As]*max_t)
            elif As.shape[2] != max_t:
                raise ValueError('Max t does not fit A')
                
            x = x0.reshape((-1,1))
            for t in range(max_t):
                x = np.hstack([x,  (As[:,:,t] @ x[:,-1].reshape((-1,1)) + offset[:,t].reshape((-1,1))).reshape((-1,1)) ])
            return x
    else:   
        dim = As.shape[0]
        if len(As.shape) == 3:
            As = As + np.eye(dim).reshape((-1,dim, 1))
        else:
            As = As + np.eye(dim).reshape((-1,dim))
        if max_t <= 0:
            if len(As.shape) == 2:
                raise ValueError('you must provide max_t if A constant')
            max_t = As.shape[2]
        if len(As.shape) == 2:
            As = np.dstack([As]*max_t)
        elif As.shape[2] != max_t:
            raise ValueError('Max t does not fit A')
            
        if checkEmptyList(offset):        
            x = x0.reshape((-1,1))
            for t in range(max_t):        
                x = np.hstack([x,  (As[:,:,t] @ x[:,-1].reshape((-1,1)) ).reshape((-1,1)) ])
            return x
        else:
            if is_1d(offset):
                offset = offset.reshape((-1,1))
                offset = np.hstack([offset]*max_t)
            if offset.shape[0] != As.shape[0]:
                raise ValueError('Offset shape does not match A shape?!')          
                
                
            if len(As.shape) == 2:
                As = np.dstack([As]*max_t)
            elif As.shape[2] != max_t:
                raise ValueError('Max t does not fit A')
                
            x = x0.reshape((-1,1))
            for t in range(max_t):
                x = np.hstack([x,  (As[:,:,t] @ x[:,-1].reshape((-1,1)) + offset[:,t].reshape((-1,1))).reshape((-1,1)) ])
            return x
            
        
    
    





def keep_thres_only(mat, thres, direction = 'lower', perc = False, num = True):
    """
    hard_threshold hard thres
    Reset to zero some elements, keep only values above/below a threshold.

    Parameters:
    - mat (numpy.ndarray): The input matrix.
    - thres (float): The threshold value. Elements below/above this value will be set to zero.
    - direction (str, optional): Direction to apply the threshold. 'lower' (default) sets elements below the threshold to zero,
      'upper' sets elements above the threshold to zero.
    - perc (bool, optional): If True, interpret thres as a percentile value. If thres is less than 1, it's treated as a percentage.
    - num (bool, optional): If True, interpret thres as the number of smallest/largest elements to keep.

    Returns:
    - numpy.ndarray: A new matrix with elements below/above the threshold set to zero.

    Raises:
    - ValueError: If both perc and num are provided, or if perc is True and thres is not in the range (0, 1).
    """    
    # reset to zero some elements, keep only perc
    # perc is percentile
    # num is how many zeros
    mat = mat.copy()
    if thres == 0:
        return mat
    if perc and num:
        raise ValueError('you must provide only perc OR  num, or neither')
    if perc and thres < 1:
        thres *= 100
        thres = np.percentile(np.abs(mat.flatten()), thres)
    if num and thres > 0:
        mat_ord = np.sort(np.abs(mat.flatten()))
        thres = mat_ord[int(thres) - 1]
        
    mat = mat.copy()
    if direction == 'lower':
        mat[np.abs(mat) <= thres] = 0 
    else:
        mat[np.abs(mat) >= thres] = 0 
    return mat














# from matplotlib.text import Annotation
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# from mpl_toolkits.mplot3d.proj3d import proj_transform
# # https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

# class Annotation3D(Annotation):

#     def __init__(self, text, xyz, *args, **kwargs):
#         super().__init__(text, xy=(0, 0), *args, **kwargs)
#         self._xyz = xyz

#     def draw(self, renderer):
#         x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
#         self.xyz = (x2, y2, z2)
#         super().draw(renderer)
     
# def _annotate3D(ax, text, xyz, *args, **kwargs):
#     '''Add anotation `text` to an `Axes3d` instance.'''
    
#     annotation = Annotation3D(text, xyz, *args, **kwargs)
#     ax.add_artist(annotation)
# setattr(Axes3D, 'annotate3D', _annotate3D)       
    
    
    
    
    # ax.annotate3D('point 2', (0, 1, 0),
    #               xytext=(-30, -30),
    #               textcoords='offset points',
    #               arrowprops=dict(ec='black', fc='white', shrink=2.5))
    
