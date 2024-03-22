import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from spatial_reconstruction import *
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
from Visualisation.qnm_vis import * 
from qnmfitsrd.CCE_file_getter import *
import datetime

sim = CCE_to_sim_high_res(sim_num='0001')

modes = [(2,2,0,1), (2,2,1,1), (2,2,2,1)] 
t0 = [20, 40, 50]

best_fit = qnmfits.ringdown_fit_dst(
    sim.times,
    sim.h[2,2],
    modes,
    Mf=sim.Mf,
    chif=sim.chif_mag,
    t0=t0
)

qnmfits.plot_mode_amplitudes(
    best_fit['C'], best_fit['mode_labels'], log=False)

plt.show()