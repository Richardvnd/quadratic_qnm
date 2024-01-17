import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from spatial_reconstruction import *
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
import seaborn as sns

l_max = 6
n_max = 6
t0 = {0:40., 1:18.5, 2:12., 3:8., 4:5.5, 5:3., 6:1.5, 7:0.}[n_max]

sim = qnmfits.SXS(ID=305, zero_time=(2,2))

QNMs = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, n_max+1)
                              for p in (-1, +1)]

lon = np.linspace(-np.pi, np.pi, 200)
lat = np.linspace(-np.pi/2, np.pi/2, 200)
Lon, Lat = np.meshgrid(lon, lat)

time_array = np.arange(-100, 75, 1)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12,8), sharex=True)

def spatial_mismatch_calculator(mapping):

    map = mapping[0]

    G = spheroidal(np.pi/2-Lat, Lon, map, l_max, sim.chif_mag)

    sm_list = []
    arg_list = []

    for time in time_array:
        best_fit = qnmfits.mapping_multimode_ringdown_fit(sim.times, 
                                                    sim.h, 
                                                    modes=QNMs.copy(),
                                                    Mf=sim.Mf,
                                                    chif=sim.chif_mag,
                                                    t0=time,
                                                    mapping_modes=mapping,
                                                    spherical_modes=[(l,m) for l in np.arange(2, l_max+1)
                                                                    for m in np.arange(-l,l+1)])

        F = mode_mapping(np.pi/2-Lat, Lon, best_fit, map, l_max)

        sm, arg, _ = spatial_mismatch(F, G, num_points=100)
        sm_list.append(sm)
        arg_list.append(arg)

    return sm_list, arg_list

map_modes = [[(2,2,0,1,2,2,0,1)], [(2,2,0,1)], [(3,2,0,1)], [(3,3,0,1)], [(4,4,0,1)]]

sm_array = []
arg_array = []
arg_unwrap_array = []
omegas = [] 

for map_mode in map_modes:
    if len(map_mode[0]) == 4:
        l, m, n, p = map_mode[0]
        omegas.append(qnmfits.qnm.omega(l,m,n,p,sim.chif_mag, Mf=sim.Mf).real)
    elif len(map_mode[0]) == 8:
        l, m, n, p, e, f, g, sign2 = map_mode[0]
        omegas.append(qnmfits.qnm.omega(l,m,n,p,sim.chif_mag, Mf=sim.Mf).real*2)    
    sm_list, arg_list = spatial_mismatch_calculator(map_mode)
    sm_array.append(sm_list)
    arg_array.append(arg_list)
    arg_unwrap_array.append(np.unwrap(arg_list))

palette = sns.color_palette("Set1", len(sm_array))

for i, mode in enumerate(map_modes):
    axs[0].plot(time_array, sm_array[i], color=palette[i], label=mode[0])
    axs[1].plot(time_array, arg_unwrap_array[i], color=palette[i], label=mode[0])
    angle_v_time = UnivariateSpline(time_array, -arg_unwrap_array[i], k=4, s=0)
    axs[2].plot(time_array, angle_v_time.derivative()(time_array), color=palette[i], label=mode[0])
    axs[2].axhline(omegas[i], ls='--', lw=1, color=palette[i], label=mode[0])

axs[0].set_yscale('log')
axs[0].set_ylabel('Spatial Mismatch')
axs[2].set_xlabel('Time')
axs[1].set_ylabel('Argument') 
axs[2].set_ylabel('Frequency') 

axs[0].legend()

plt.savefig('lobe_frequency.png')
plt.show()
