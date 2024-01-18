
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

l_max = 3
n_max = 3
t_start = -1
t_end = 1
t_step = 1

mapping_QNMs = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, n_max+1)
                              for p in (-1, +1)]

mapping = [(2,2,0,1)] 

model_modes = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, n_max+1)
                              for p in (-1, +1)]

id = datetime.datetime.now()
sim = qnmfits.SXS(ID=305, zero_time=(2,2))

ani = qnm_vis.animate_mapping_qnm_lm(sim, 
                                     tstart=-200, 
                                     tstop=200, 
                                     tstep = 1, 
                                     mapping_QNMs=mapping_QNMs, 
                                     mapping=mapping, 
                                     mapping_spherical_modes=None, 
                                     l_max=l_max,
                                     data_spherical_modes=None, 
                                     model_modes=None, 
                                     projection='mollweide')

ani.save(f'full_{id}.mp4', writer='ffmpeg')