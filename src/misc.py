from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from decimal import Decimal
from scipy.optimize import linear_sum_assignment

"""
This script contains all kinds of functions
"""

def bools2str(x):
    obaxes = ''
    for i in(x):
        if i:
            obaxes+='1'
        else:
            obaxes+='0'
    return obaxes

# polygon angle
THETA = 2*np.pi/360*150
# POLYGONSIZE/size of polygons
POLYGONSIZE = 0.1


def get_cmap(n: int, cmapname:str = 'hsv') -> Colormap:
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmapname, n)

def n_colors(n: int, cmapname:str = 'hsv') -> List:
    '''generates list list of n colors from given matplotlib colormap'''
    cmap = get_cmap(n, cmapname)
    return [cmap(i) for i in range(n)]


def xyphi_to_abc(x: float, y: float, phi: float) -> np.ndarray:
    '''converts x, y, phi to triangle points a, b, c'''
    ax = (x+POLYGONSIZE*np.cos(phi))
    ay = (y+POLYGONSIZE*np.sin(phi))

    bx = (x+POLYGONSIZE*np.cos(phi+THETA))
    by = (y+POLYGONSIZE*np.sin(phi+THETA))

    cx = (x+POLYGONSIZE*np.cos(phi-THETA))
    cy = (y+POLYGONSIZE*np.sin(phi-THETA))

    triangle = np.array([[ax, ay], [bx, by], [cx, cy]])
    return triangle

def format_e(n):
    return "{:0.1f}%".format(n*100)


###### MATHS

def periodic_distant_vectors(vectors, box_size):
    
    assert vectors.shape[-1] == 2, f"Dont pass angles in this function you dumbo!, { vectors.shape[-1]}"
    
    vectors[:,:,0:2] = np.where(vectors[:,:,0:2]>box_size/2,vectors[:,:,0:2]-box_size,vectors[:,:,0:2])
    vectors[:,:,0:2] = np.where(vectors[:,:,0:2]<-box_size/2,vectors[:,:,0:2]+box_size,vectors[:,:,0:2])

    
    return vectors


def absolute_distances_with_periodic_box_size(
    a_positions,
    b_positions,        
    box_size,
):
    _a1 = np.tile(a_positions, (a_positions.shape[0], 1, 1))
    _b1 = np.tile(b_positions, (b_positions.shape[0], 1, 1))
    _b1 = np.transpose(_b1, axes=(1, 0, 2))
    distances = _a1-_b1
    
    distances = periodic_distant_vectors(distances, box_size)
        
    return np.linalg.norm(distances, axis=2)

def foldback_dist_ensemble(ensemble_a, ensemble_b, box_size):
    vectors = ensemble_a - ensemble_b
    vectors[:,:,0:2] = np.where(vectors[:,:,0:2]>box_size/2,vectors[:,:,0:2]-box_size,vectors[:,:,0:2])
    vectors[:,:,0:2] = np.where(vectors[:,:,0:2]<-box_size/2,vectors[:,:,0:2]+box_size,vectors[:,:,0:2])
    
    vectors[:,:,3] = np.angle(np.exp(1j* (ensemble_a[:,:,3] - ensemble_b[:,:,3]) ))
    
    return vectors
    


def foldback_dist_states(state_a, state_b, box_size, theta_axis=-1):
    
    vectors = state_a - state_b

    vectors[:,0:2] = np.where(vectors[:,0:2]>box_size/2,vectors[:,0:2]-box_size,vectors[:,0:2])
    vectors[:,0:2] = np.where(vectors[:,0:2]<-box_size/2,vectors[:,0:2]+box_size,vectors[:,0:2])
    
    # Theta axis is false or -1
    if theta_axis:
        vectors[:,-1] = np.mod(state_a[:,-1] - state_b[:,-1] + np.pi, 2*np.pi) - np.pi
    
    return vectors

def mean_over_ensemble(ensemble, box_size):
    
    p1 = ensemble[0,:,:]
    n_ensemble = ensemble.shape[0]
    dists = ensemble - p1

    # apply pbc to dist
    dists[:,:,0:2] = periodic_distant_vectors(dists[:,:,0:2], box_size)

    average = p1 + 1/(n_ensemble) * np.sum(dists, axis=0)

    average[:,0] = np.mod(average[:,0], box_size)
    average[:,1] = np.mod(average[:,1], box_size)
    average[:,2] = np.mean(ensemble[:,:,2], axis=0)
    average[:,3] = np.angle(np.sum( np.exp(1j* ensemble[:,:,3] ) ,axis=0))
        
    
    return average





def assign_fn(measurement_positions, state_positions, box_size):
    rowids, colids = linear_sum_assignment(
        absolute_distances_with_periodic_box_size(measurement_positions, state_positions, box_size=box_size)
    )
    return colids


###### METRICS

def metric_hungarian_precision(viscek_positions: np.ndarray, kalman_positions: np.ndarray, box_size:float) -> float:
    n_particles = np.shape(viscek_positions)[0]
    cost_matrix = absolute_distances_with_periodic_box_size(viscek_positions, kalman_positions, box_size=box_size)
    rowid, col_id = linear_sum_assignment(cost_matrix)
    precision = 1/n_particles*sum(np.diag(np.ones(n_particles))[rowid, col_id])
    return precision

def metric_lost_particles(viscek_positions: np.ndarray, kalman_positions: np.ndarray, dist_thresh: float) -> float:
    n_particles = np.shape(viscek_positions)[0]
    abs_distances = np.linalg.norm(kalman_positions - viscek_positions, axis=1)
    right_particles =  sum(abs_distances <= dist_thresh)
    return right_particles/n_particles
    
    
def metric_flocking(viscek_angles: np.ndarray, n_particles: float) -> float:
    v_xtot = np.sum(np.cos(viscek_angles))
    v_ytot = np.sum(np.sin(viscek_angles))
    return np.sqrt(v_xtot**2 + v_ytot**2) / n_particles