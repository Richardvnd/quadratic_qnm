import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from spatial_reconstruction_tests.spatial_reconstruction import *
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
from Visualisation.qnm_vis import * 
from qnmfitsrd.CCE_file_getter import *
import datetime

l_max = 5
n_max = 5

sim = CCE_to_sim_simplified(sim_num='0305')
#sim = qnmfits.SXS(ID=305, zero_time=(2,2))

quadratic_mode = [(2,2,0,1,2,2,0,1)]

l_modes = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, n_max+1)
                              for p in (-1, +1)]

lq_modes = l_modes + quadratic_mode

best_fit = qnmfits.multimode_ringdown_fit(sim.times, 
                                 sim.h, 
                                 modes=lq_modes,
                                 Mf=sim.Mf,
                                 chif=sim.chif_mag,
                                 t0=0)