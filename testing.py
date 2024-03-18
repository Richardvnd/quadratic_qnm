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

sim = CCE_to_sim_simplified(sim_num='0001')
simhr = CCE_to_sim_high_res(sim_num='0001')

plt.plot(sim.times, sim.h[2,2].real)
plt.plot(simhr.times, simhr.h[2,2].real, 'r')

plt.xlim(0, 50)

plt.show()