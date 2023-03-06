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
    L,
):
    max_distance = math.sqrt(2*L**2)/2
    distances = distance_matrix(a_positions, b_positions)
    bad_dists_idxs = np.where(distances>max_distance)
    # iterate over all bad pairs
    for a_idx, b_idx in zip(bad_dists_idxs[0], bad_dists_idxs[1]):
        a = a_positions[a_idx]
        b = b_positions[b_idx]
        
        ax = a[0]
        ay = a[1]
        bx = b[0]
        by = b[1]

        if abs(ax-bx) > abs(ay-by):
            #right-left shift
            _dy = ay-by
            ynew = by
            if ax < bx:
                # print('shift b over right')
                xnew = bx-L
            else:
                # print('shift b over left')
                xnew = bx+L
            _dx = ax-xnew
        else:
            #top-down shift
            _dx = ax-bx
            xnew = bx
            if ay < by:
                # print('shift b over top')
                ynew = by-L
            else:
                # print('shift b over bottom')
                ynew = by+L

            _dy = ay-ynew
        distances[a_idx][b_idx] = math.sqrt(
            _dx**2+_dy**2
        )
    return distances

def assign_fn(measurement_positions, state_positions, boundary):
    rowids, colids = linear_sum_assignment(
        distances_with_periodic_boundary(measurement_positions, state_positions, L=boundary)
    )
    return colids

def metric_hungarian_precision(viscek_positions: np.ndarray, kalman_positions: np.ndarray, boundary:float) -> float:
    n_particles = np.shape(viscek_positions)[0]
    cost_matrix = distances_with_periodic_boundary(viscek_positions, kalman_positions, L=boundary)
    rowid, col_id = linear_sum_assignment(cost_matrix)
    precision = 1/n_particles*sum(np.diag(np.ones(n_particles))[rowid, col_id])
    return precision

def metric_lost_particles(viscek_positions: np.ndarray, kalman_positions: np.ndarray, dist_thresh: float) -> float:
    n_particles = np.shape(viscek_positions)[0]
    abs_distances = np.linalg.norm(kalman_positions - viscek_positions, axis=1)
    right_particles =  sum(abs_distances <= dist_thresh)
    return right_particles/n_particles
    
    
def metric_flocking(viscek_angles: np.ndarray, velocity, n_particles: float) -> float:
    v_xtot = np.sum(velocity*np.cos(viscek_angles))
    v_ytot = np.sum(velocity*np.sin(viscek_angles))
    return np.sqrt(v_xtot**2 + v_ytot**2) / n_particles / velocity