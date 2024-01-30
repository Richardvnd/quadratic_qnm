"""

Updated: 05/01/2024

This generates animations of the spatial reconstruction for a given mapping. It also plots the mismatch,
and argument of the <f|g*>. 

"""

import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from spatial_reconstruction import *
from matplotlib.animation import FuncAnimation
import datetime 

id = 305

sim = qnmfits.SXS(ID=id, zero_time=(2,2))

l_max = 3
n_max = 3
t_start = -100
t_end = 200
t_step = 1
t0 = {0:40., 1:18.5, 2:12., 3:8., 4:5.5, 5:3., 6:1.5, 7:0.}[n_max]
time_id = datetime.datetime.now()

mapping = [(2,2,0,1)]

QNMs = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, n_max+1)
                              for p in (-1, +1)]

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,10))

axs[0, 0].axis('off')  
axs[0, 1].axis('off')  
axs[1, 0].axis('off')  
axs[1, 1].axis('off')  
axs[0, 0] = plt.subplot(4, 2, 1, projection='mollweide')
axs[0, 1] = plt.subplot(4, 2, 2, projection='mollweide')
axs[1, 0] = plt.subplot(4, 2, 3, projection='mollweide')
axs[1, 1] = plt.subplot(4, 2, 4, projection='mollweide')

lon = np.linspace(-np.pi, np.pi, 200)
lat = np.linspace(-np.pi/2, np.pi/2, 200)
Lon, Lat = np.meshgrid(lon, lat)

map = mapping[0]

G = spheroidal(np.pi/2-Lat, Lon, map, l_max, sim.chif_mag)

sm_list = [] 
arg_list = [] 
times_list = [] 

def update(step):

   times_list.append(step)

   best_fit = qnmfits.mapping_multimode_ringdown_fit(sim.times, 
                                             sim.h, 
                                             modes=QNMs.copy(),
                                             Mf=sim.Mf,
                                             chif=sim.chif_mag,
                                             t0=step,
                                             mapping_modes=mapping,
                                             spherical_modes=[(l,m) for l in np.arange(2, l_max+1)
                                                               for m in np.arange(-l,l+1)])

   F = mode_mapping(np.pi/2-Lat, Lon, best_fit, map, l_max)
   sm, arg, _ = spatial_mismatch(F, G, num_points=100)
   sm_list.append(sm)
   arg_list.append(arg)

   fig.suptitle(f"Map: {map} \n $t_0$ = {step}", fontsize=16)

   axs[0, 0].title.set_text('Real reconstruction')
   axs[0, 0].title.set_position([0.5, 1.05])  
   axs[0, 0].pcolormesh(Lon, Lat, np.real(F), cmap=plt.cm.jet)

   axs[1,0].title.set_text('Imag reconstruction')
   axs[1,0].title.set_position([0.5, 1.05])  
   axs[1,0].pcolormesh(Lon, Lat, np.imag(F), cmap=plt.cm.jet)

   axs[0, 1].title.set_text('Real expected')
   axs[0, 1].pcolormesh(Lon, Lat, np.real(G), cmap=plt.cm.jet)

   axs[1,1].title.set_text('Imag expected')
   axs[1,1].pcolormesh(Lon, Lat, np.imag(G), cmap=plt.cm.jet)

   axs[2, 0].plot(times_list, sm_list, color='black')
   axs[2, 0].set_ylabel('Spatial Mismatch')
   axs[2, 0].set_xlabel('Time')

   axs[2,1].set_ylabel('SM argument (unwrapped)')
   axs[2,1].title.set_position([0.5, 1.05])  
   axs[2,1].plot(times_list, np.unwrap(arg_list), color='black')
   axs[2,1].set_xlabel('Time')

   return fig 

ani = FuncAnimation(fig, update, frames=range(t_start, t_end, t_step), interval=100)
ani.save(f'mapping_animation_{map}_{id}_{time_id}.mp4', writer='ffmpeg')