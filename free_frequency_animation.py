import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from scipy.interpolate import UnivariateSpline
from qnmfitsrd.CCE_file_getter import *
from scipy.optimize import least_squares

sims = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013']
QNMs = [(3,3,0,1,2,-2,0,-1), (2,2,0,1,2,-2,0,-1),(2,2,0,1,2,2,0,1), (3,3,0,1,2,2,0,1)] 

t_start = -20
t_end = 30 
t_step = 1

for quadratic_mode in QNMs:

    l1,m1,n1,p1,l2,m2,n2,p2 = quadratic_mode 
    lp = l1 + l2 
    mp = m1 + m2

    if mp < 2:
        lp_min = 2
    else:
        lp_min = mp

    spherical_mode = (lp_min, mp)

    model_q = [(lp_min,mp,n,1) for n in range(1+1)] + [(lp_min,mp,n,-1) for n in range(1+1)] + [quadratic_mode]

    re_min = -1.2
    re_max = 1.7
    im_min = -0.6
    im_max = 0.25
    step = 101 

    index = model_q.index(quadratic_mode)

    im_variation = np.linspace(im_min, im_max, step)
    re_variation = np.linspace(re_min, re_max, step)

    for simnum in sims:

        sim = CCE_to_sim_high_res(sim_num=simnum)

        frequencies = np.array(qnmfits.qnm.omega_list(model_q, sim.chif_mag, sim.Mf))

        fig, ax = plt.subplots() 

        xmin = re_min
        xmax = re_max
        ymin = im_min
        ymax = im_max

        ax.set_xlabel(f'Re($\omega_{{{quadratic_mode}}}^{{{spherical_mode}}}$)')
        ax.set_ylabel(f'Im($\omega_{{{quadratic_mode}}}^{{{spherical_mode}}}$)')

        ax.plot(frequencies[index].real, frequencies[index].imag, color='r', marker='x', markersize=10)

        """        
        for k, freq in enumerate(frequencies):
                    if freq != frequencies[index]:
                        ax.plot(freq.real, freq.imag, color='w', marker='o', markersize=5)
                        ax.text(freq.real, freq.imag+0.05, f'{model_q[k]}', fontsize=5, color='w')
                        ax.plot(-freq.real, freq.imag, color='w', marker='o', markersize=5)
                        ax.text(freq.real, freq.imag+0.05, f'{model_q[k]}', fontsize=5, color='w')
                        """

        next_freq = qnmfits.qnm.omega(lp_min,mp,2,1, sim.chif_mag, Mf=sim.Mf)
        ax.plot(next_freq.real, next_freq.imag, color='r', marker='o', markersize=5)
        ax.text(next_freq.real, next_freq.imag+0.05, f'({lp},{mp},2,1)', fontsize=5, color='r')

        def update(t0): 

            mm_list = [] 

            for v_im in im_variation:
                for v_re in re_variation:

                    modified_frequencies = frequencies.copy()

                    modified_frequencies[index] = v_re + 1j * v_im

                    best_fit = qnmfits.ringdown_fit(
                        sim.times,
                        sim.h[lp_min,mp],
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

        savepath = "free_frequency_fits"

        ani = FuncAnimation(fig, update, frames=range(t_start, t_end, t_step), interval=100)
        ani.save(f'{savepath}/free_frequency_{quadratic_mode}_{simnum}.mp4', writer='ffmpeg')