
"""
Updated: 17/01/24

This code is intended to generate qnm animations using the qnm_vis class.

"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from qnm_vis import *

qnm_vis = qnm_vis() 

id = datetime.datetime.now()
sim = qnmfits.SXS(ID=305, zero_time=(2,2))
ani  = qnm_vis.animate_spherical_modes(sim.times, sim.h, 
                                tstart = -1800, tstop = 500, tstep = 5, projection='mollweide')
ani.save(f'lm_{id}.mp4', writer='ffmpeg')