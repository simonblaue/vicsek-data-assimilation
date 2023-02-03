from dataclasses import dataclass, field
import numpy as np

# simulation steps made before animation is updated
FREQUENCY = 2

# simulation steps made before giving the data to the filter
RESOLUTION = 1

# polygon angle
THETA = 2*np.pi/360*150
# POLYGONSIZE/size of polygons
POLYGONSIZE = 0.1
