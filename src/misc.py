from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from decimal import Decimal
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

"""
This script contains all kinds of functions
"""

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
    return "{:.2E}".format(Decimal(n))

def metric_hungarian_precision(viscek_positions: np.ndarray, kalman_positions: np.ndarray, n_particles: int) -> float:
    cost_matrix = distance_matrix(viscek_positions, kalman_positions)
    rowid, col_id = linear_sum_assignment(cost_matrix)
    precision = 1/n_particles*sum(np.diag(np.ones(n_particles))[rowid, col_id])
    return precision

def metric_lost_particles(viscek_positions: np.ndarray, kalman_positions: np.ndarray, n_particles: int, dist_thresh: float) -> float:
    abs_distances = np.linalg.norm(kalman_positions - viscek_positions, axis=1)
    lost_particles =  sum(abs_distances <= dist_thresh)
    return lost_particles/n_particles
    