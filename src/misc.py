from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from decimal import Decimal
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import math


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

def distances_with_periodic_boundary(
    a_positions,
    b_positions,        
    boundary,
):
    _a1 = np.tile(a_positions, (a_positions.shape[0], 1, 1))
    _b1 = np.tile(b_positions, (b_positions.shape[0], 1, 1))
    _b1 = np.transpose(_b1, axes=(1, 0, 2))
    distances = _a1-_b1
    
    distances[:,:,0] = np.where(distances[:,:,0]>boundary/2,distances[:,:,0]-boundary,distances[:,:,0])
    distances[:,:,0] = np.where(distances[:,:,0]<-boundary/2,distances[:,:,0]+boundary,distances[:,:,0])
        
    distances[:,:,1] = np.where(distances[:,:,1]>boundary/2,distances[:,:,1]-boundary,distances[:,:,1])
    distances[:,:,1] = np.where(distances[:,:,1]<-boundary/2,distances[:,:,1]+boundary,distances[:,:,1])
    return np.linalg.norm(distances, axis=2)

def assign_fn(measurement_positions, state_positions, boundary):
    rowids, colids = linear_sum_assignment(
        distances_with_periodic_boundary(measurement_positions, state_positions, boundary=boundary)
    )
    return colids

def metric_hungarian_precision(viscek_positions: np.ndarray, kalman_positions: np.ndarray, boundary:float) -> float:
    n_particles = np.shape(viscek_positions)[0]
    cost_matrix = distances_with_periodic_boundary(viscek_positions, kalman_positions, boundary=boundary)
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