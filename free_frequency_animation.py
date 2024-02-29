import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
from qnmfitsrd.CCE_file_getter import *
import datetime
from scipy.optimize import least_squares

t_start = -50
t_end = 50 
t_step = 1

CCE_sim305 = CCE_to_sim_simplified(sim_num='0305')

quadratic_mode = (2,2,0,1,2,-2,0,-1) 
spherical_mode = (4,0) 
sim = CCE_sim305
#CCE_sim12

l,m = spherical_mode 
lp,mp,nprime,pp,lpp,mpp,npp,ppp = quadratic_mode

model_q = [(l,m,n,1) for n in range(1+1)] + [(l,m,n,-1) for n in range(1+1)] + [quadratic_mode]

re_min = -1.2
re_max = 1.7
im_min = -0.6
im_max = 0.25
step = 101 

time_id = datetime.datetime.now()

index = model_q.index(quadratic_mode)

im_variation = np.linspace(im_min, im_max, step)
re_variation = np.linspace(re_min, re_max, step)

frequencies = np.array(qnmfits.qnm.omega_list(model_q, sim.chif_mag, sim.Mf))

fig, ax = plt.subplots() 

xmin = re_min
xmax = re_max
ymin = im_min
ymax = im_max

ax.set_xlabel(f'Re($\omega_{{{quadratic_mode}}}^{{{spherical_mode}}}$)')
ax.set_ylabel(f'Im($\omega_{{{quadratic_mode}}}^{{{spherical_mode}}}$)')

ax.plot(frequencies[index].real, frequencies[index].imag, color='r', marker='x', markersize=10)

for i, freq in enumerate(frequencies):
    if freq != frequencies[index]:
        ax.plot(freq.real, freq.imag, color='w', marker='o', markersize=5)
        ax.text(freq.real, freq.imag+0.05, f'{model_q[i]}', fontsize=5, color='w')
        ax.plot(-freq.real, freq.imag, color='w', marker='o', markersize=5)
        ax.text(freq.real, freq.imag+0.05, f'{model_q[i]}', fontsize=5, color='w')

next_freq = qnmfits.qnm.omega(l,m,2,1, sim.chif_mag, Mf=sim.Mf)
ax.plot(next_freq.real, next_freq.imag, color='r', marker='o', markersize=5)
ax.text(next_freq.real, next_freq.imag+0.05, f'({l},{m},2,1)', fontsize=5, color='r')

def update(t0): 

    mm_list = [] 

    for v_im in im_variation:
        for v_re in re_variation:

            modified_frequencies = frequencies.copy()

            modified_frequencies[index] = v_re + 1j * v_im

            best_fit = qnmfits.ringdown_fit(
                sim.times,
                sim.h[l,m],
                model_q,
                Mf=sim.Mf,
                chif=sim.chif_mag,
                t0=t0,
                frequencies = modified_frequencies
            )

            mm_list.append(best_fit['mismatch'])

    mm_grid = np.reshape(
        np.array(mm_list), (step, step)
        )

    im = ax.imshow(
        np.log10(mm_grid), 
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        interpolation='bicubic',
        cmap='viridis_r')

    ax.set_title(f't = {t0}')

    return fig 

ani = FuncAnimation(fig, update, frames=range(t_start, t_end, t_step), interval=100)
ani.save(f'free_frequency_{quadratic_mode}_{time_id}.mp4', writer='ffmpeg')