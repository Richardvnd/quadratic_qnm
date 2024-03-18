import numpy as np
import matplotlib.pyplot as plt
import qnmfitsrd as qnmfits
from multiprocessing import Pool
from spatial_reconstruction import *
import matplotlib.pyplot as plt
from qnmfitsrd.CCE_file_getter import *

sims = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013']
QNMs = [[(2,2,1,1,2,2,1,1)], [(3,3,1,1,2,2,1,1)], [(3,3,1,1,2,-2,1,-1)], [(2,2,1,1,2,-2,1,-1)]] 

l_max = 8
n_max = 3

start = -30
stop = 30

for mapping in QNMs: 
    
    map = mapping[0]
    l1,m1,n1,p1,l2,m2,n2,p2 = map 
    i = l1 + l2 
    j = m1 + m2

    if j < 2:
        l_start = 2
    else:
        l_start = j

    for simnum in sims: 

        sim = CCE_to_sim_high_res(sim_num=simnum)

        QNMs = [(lam, j, n, p) for lam in np.arange(l_start, l_max+1) for n in np.arange(0,n_max+1) for p in (-1, +1)]

        spherical_modes = [(l,j) for l in range(l_start,l_max+1)]

        decay_time = qnmfits.qnm.omega(l1,m1,n1,p1, sim.chif_mag, Mf=sim.Mf).imag + qnmfits.qnm.omega(l2,m2,n2,p2, sim.chif_mag, Mf=sim.Mf).imag

        amplitudes1 = []
        amplitudes2 = []
        amplitudes3 = []

        sphindex1 = spherical_modes.index((l_start,j))
        sphindex2 = spherical_modes.index((l_start+1,j))
        sphindex3 = spherical_modes.index((l_start+2,j))

        #print(f"Mapping {map} to {spherical_modes} with model {QNMs} and testing amplitudes of {l_start}{j}, {l_start+1}{j}, and {l_start+2}{j}.")

        for step in range(start,stop):

            best_fit = qnmfits.mapping_multimode_ringdown_fit(sim.times, 
                                                            sim.h,
                                                            modes=QNMs.copy(),
                                                            Mf=sim.Mf,
                                                            chif=sim.chif_mag,
                                                            t0=step,
                                                            mapping_modes=mapping,
                                                            spherical_modes=spherical_modes) 
            
            # NEED A -1 IF MAPPING QNM INCLUDED IN MODEL 

            amplitudes1.append(best_fit['C'][len(QNMs) + sphindex1])
            amplitudes2.append(best_fit['C'][len(QNMs) + sphindex2])
            amplitudes3.append(best_fit['C'][len(QNMs) + sphindex3])

        mu20 = qnmfits.qnm.alpha([(l_start,j)+map], sim.chif_mag)[0]
        mu30 = qnmfits.qnm.alpha([(l_start+1,j)+map], sim.chif_mag)[0]
        mu40 = qnmfits.qnm.alpha([(l_start+2,j)+map], sim.chif_mag)[0]

        mu20a = qnmfits.qnm.alternative_alpha([(l_start,j)+map], sim.chif_mag)
        mu30a = qnmfits.qnm.alternative_alpha([(l_start+1,j)+map], sim.chif_mag)
        mu40a = qnmfits.qnm.alternative_alpha([(l_start+2,j)+map], sim.chif_mag)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8)) 

        ax[0,0].plot(np.arange(start,stop), np.real(amplitudes1), label=f'{l_start}{j} real')
        ax[0,0].plot(np.arange(start,stop), np.imag(amplitudes1), label=f'{l_start}{j} imag')
        ax[0,0].plot(np.arange(start,stop), np.real(amplitudes2), label=f'{l_start+1}{j} real')
        ax[0,0].plot(np.arange(start,stop), np.imag(amplitudes2), label=f'{l_start+1}{j} imag')
        ax[0,1].plot(np.arange(start,stop), np.real(amplitudes3), label=f'{l_start+2}{j} real')
        ax[0,1].plot(np.arange(start,stop), np.imag(amplitudes3), label=f'{l_start+2}{j} imag')

        ratio1 = [abs(a / b) for a, b in zip(amplitudes1, amplitudes2)]
        ax[1,0].plot(np.arange(start,stop), ratio1, label=f'Abs mapped {l_start}{j}/{l_start+1}{j}')

        ratio2 = [abs(a / b) for a, b in zip(amplitudes1, amplitudes3)]
        ax[1,1].plot(np.arange(start,stop), ratio2, label=f'Abs mapped {l_start}{j}/{l_start+2}{j}')

        ax[1,0].axhline(y=abs(mu20/mu30), color='black', linestyle='--', label=f'Expected {l_start}{j}/{l_start+1}{j}')
        ax[1,1].axhline(y=abs(mu20/mu40), color='black', linestyle='--', label=f'Expected {l_start}{j}/{l_start+2}{j}')

        ax[1,0].set_title(f'{l_start}{j}/{l_start+1}{j}')
        ax[1,1].set_title(f'{l_start}{j}/{l_start+2}{j}')

        ax[0,1].legend()
        ax[0,0].legend()
        ax[1,0].legend()
        ax[1,1].legend()

        #ax[0,0].set_yscale('log')
        #ax[0,1].set_yscale('log')
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
        
        plt.suptitle(f'Sim: {simnum}')

        plt.savefig(f'quadratic_mapping_charts/qm_{map}_{simnum}.png')