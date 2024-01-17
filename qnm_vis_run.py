
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

modes = [(3,2,0,1), (3,3,0,1), (3,1,0,1), (3,0,0,1)]

id = datetime.datetime.now()
sim = qnmfits.SXS(ID=305, zero_time=(2,2))
ani  = qnm_vis.animate_qnms(sim, modes=modes, tstart=0, tend=20)
ani.save(f'spherical_{id}.mp4', writer='ffmpeg')