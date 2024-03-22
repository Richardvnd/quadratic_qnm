import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
from qnmfitsrd.CCE_file_getter import *
from scipy.optimize import least_squares

simnum = '0001'
target_mode = (2,2,0,1,2,-2,0,-1)
spherical_mode = (2,0)
model = [(2,0,0,1), (2,0,0,-1)] + [target_mode]


#l,m,n,p = target_mode
lp, mp = spherical_mode
target_index = model.index(target_mode)

sim = CCE_to_sim_high_res(sim_num=simnum)

t_start = -20
t_end = 100 
t_step = 1

re_min = -1.2
re_max = 1.7
im_min = -0.6
im_max = 0.25
step = 101 

im_variation = np.linspace(im_min, im_max, step)
re_variation = np.linspace(re_min, re_max, step)

target_frequency = np.array(qnmfits.qnm.omega_list(model, sim.chif_mag, sim.Mf))

fig, ax = plt.subplots() 

xmin = re_min
xmax = re_max
ymin = im_min
ymax = im_max

ax.set_xlabel(f'Re($\omega_{{{target_mode}}}^{{{spherical_mode}}}$)')
ax.set_ylabel(f'Im($\omega_{{{target_mode}}}^{{{spherical_mode}}}$)')

ax.plot(target_frequency[target_index].real, target_frequency[target_index].imag, color='r', marker='x', markersize=10)

#next_freq = qnmfits.qnm.omega(l,m,n,-1, sim.chif_mag, Mf=sim.Mf)
#ax.plot(next_freq.real, next_freq.imag, color='r', marker='o', markersize=5)
#ax.text(next_freq.real, next_freq.imag+0.05, f'({l},{m},0,-1)', fontsize=5, color='r')

def update(t0): 

    mm_list = [] 

    for v_im in im_variation:
        for v_re in re_variation:

            modified_frequency = target_frequency.copy()

            modified_frequency[target_index] = v_re + 1j * v_im

            best_fit = qnmfits.ringdown_fit(
                sim.times,
                sim.h[lp,mp],
                model,
                Mf=sim.Mf,
                chif=sim.chif_mag,
                t0=t0,
                frequencies = modified_frequency
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

savepath = "free_frequency_fits"

ani = FuncAnimation(fig, update, frames=range(t_start, t_end, t_step), interval=100)
ani.save(f'free_frequency_{target_mode}_{spherical_mode}_{simnum}_nomodes.mp4', writer='ffmpeg')