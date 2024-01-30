import datetime
import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from Visualisation.qnm_vis import *

qnm_vis = qnm_vis() 

modes = [(2,0,0,1)]

id = datetime.datetime.now()
sim = qnmfits.SXS(ID=305, zero_time=(2,2))
ani  = qnm_vis.animate_qnms(sim, modes=modes, tstart=-50, tend=100)
ani.save(f'spherical_{id}.mp4', writer='ffmpeg')