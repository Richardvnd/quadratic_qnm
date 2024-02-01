"""
Updated: 16/01/2024

A class of functions for visualising the spherical modes and qnms spatially and over time (as an animation).

"""

import numpy as np 
import matplotlib.pyplot as plt 
import qnmfitsrd.qnm as qnm
from matplotlib.animation import FuncAnimation
import quaternionic
import spherical
import qnmfitsrd as qnmfits
from tqdm import tqdm 
from spatial_reconstruction_tests.spatial_reconstruction import *

class qnm_vis:
    def __init__(self, l_max=9, precomp_sYlm = True):

        self.l_max = l_max

        theta_vals = np.linspace(-np.pi, np.pi, 200)
        phi_vals = np.linspace(-np.pi/2, np.pi/2, 200)
        self.theta, self.phi = np.meshgrid(theta_vals, phi_vals) 
        
        self.x = np.sin(self.theta) * np.cos(self.phi)
        self.y = np.sin(self.theta) * np.sin(self.phi)
        self.z = np.cos(self.theta)

        self.wigner = spherical.Wigner(l_max)

        if precomp_sYlm:
            sYlm = np.empty((l_max+1, 2*l_max+1, self.theta.shape[0], self.theta.shape[1]), dtype=complex)
            all_mesh_sYlm = self.mesh_sYlm(np.pi/2 - self.phi, self.theta, s=-2)
            for l in range(2, l_max+1):
                for m in range(-l, l+1):
                    sYlm[l,m] = all_mesh_sYlm[:,:,self.wigner.Yindex(l, m)]
            self.sYlm = sYlm


    def mesh_sYlm(self, theta, phi, s=-2):
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        Y = self.wigner.sYlm(s, R)
        return Y


    def plot_spherical_modes(self, ax, times, data_dict, theta, phi, spherical_modes, t0=0):

        t0 = min(times, key=lambda x: abs(x - t0))
        data_mask = (times == t0) 

        breakpoint() 

        map = 0 
        for lm in spherical_modes:
            map += data_dict[lm][data_mask] * self.sYlm[lm[0], lm[1]]
        map /= np.max(np.abs(map))

        ax[0].title.set_text('Real')
        ax[0].pcolormesh(theta, phi, np.real(map), cmap=plt.cm.jet)

        ax[1].title.set_text('Imaginary')
        ax[1].pcolormesh(theta, phi, np.imag(map), cmap=plt.cm.jet)

        return ax
    

    def animate_spherical_modes(self, times, data_dict, spherical_modes=None, 
                                tstart = -100, tstop = 100, tstep = 50, projection='mollweide'): 
        
        if spherical_modes is None:
            spherical_modes = list(data_dict.keys())

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
        ax[0].axis('off')  
        ax[1].axis('off')  
        ax[0] = plt.subplot(1, 2, 1, projection=projection)
        ax[1] = plt.subplot(1, 2, 2, projection=projection)

        theta, phi = self.theta, self.phi

        pbar = tqdm(total=len(range(tstart, tstop, tstep)))

        def update(step):
            self.plot_spherical_modes(ax, times, data_dict, theta, phi, spherical_modes, step)
            pbar.update(1)
            if step == tstop - tstep:
                pbar.close()
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tstop, tstep), interval=100)

        return ani
    

    def animate_qnms(self, sim, tstart=-100, tstop=200, modes=[(2,2,0,1)], tstep = 1, projection='mollweide'):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
        ax[0].axis('off')  
        ax[1].axis('off')  
        ax[0] = plt.subplot(1, 2, 1, projection=projection)
        ax[1] = plt.subplot(1, 2, 2, projection=projection)

        theta, phi = self.theta, self.phi

        spherical_modes = sim.h.keys()

        pbar = tqdm(total=len(range(tstart, tstop, tstep)))

        def update(step):
            best_fit = qnmfits.multimode_ringdown_fit(sim.times, 
                                            sim.h, 
                                            modes=modes,
                                            Mf=sim.Mf,
                                            chif=sim.chif_mag,
                                            t0=step)

            t0 = min(best_fit['model_times'], key=lambda x: abs(x - step))
            data_mask = (best_fit['model_times'] == t0) 

            map = sum(sum(best_fit['model'][lm][data_mask]) * self.sYlm[lm[0], lm[1]] for lm in spherical_modes)
            map /= np.max(np.abs(map))

            ax[0].title.set_text('Real')
            ax[0].pcolormesh(theta, phi, np.real(map), cmap=plt.cm.viridis)

            ax[1].title.set_text('Imaginary')
            ax[1].pcolormesh(theta, phi, np.imag(map), cmap=plt.cm.viridis)

            pbar.update(1)
            if step == tstop - tstep:
                pbar.close()
            
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tstop, tstep), interval=50)

        return ani
    

    def animate_mapping(self, sim, tstart=-100, tstop=200, tstep = 1, 
                               mapping_QNMs=[(2,2,0,1)], mapping=[(2,2,0,1)], l_max=9,
                               mapping_spherical_modes=None, 
                               projection='mollweide'):

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
        ax[0,0].axis('off')  
        ax[0,1].axis('off')  
        ax[1,0].axis('off')  
        ax[1,1].axis('off')  
        ax[0,0] = plt.subplot(3, 2, 1, projection=projection)
        ax[0,1] = plt.subplot(3, 2, 2, projection=projection)
        ax[1,0] = plt.subplot(3, 2, 1, projection=projection)
        ax[1,1] = plt.subplot(3, 2, 2, projection=projection)

        theta, phi = self.theta, self.phi
        
        if mapping_spherical_modes is None:
            mapping_spherical_modes = [(l,m) for l in np.arange(2, l_max+1)
                                                            for m in np.arange(-l,l+1)]
        times = sim.times
        data_dict = sim.h

        pbar = tqdm(total=len(range(tstart, tstop, tstep)))
        
        G = spheroidal(np.pi/2-phi, theta, mapping[0], l_max, sim.chif_mag)

        sm_list = [] 
        times_list = []

        def update(step):

            best_fit = qnmfits.mapping_multimode_ringdown_fit(times, 
                                            data_dict, 
                                            modes=mapping_QNMs,
                                            Mf=sim.Mf,
                                            chif=sim.chif_mag,
                                            t0=step,
                                            mapping_modes=mapping,
                                            spherical_modes=mapping_spherical_modes)

            F = mode_mapping(np.pi/2-phi, theta, best_fit, mapping[0], l_max)

            sm, _, _ = spatial_mismatch(F, G, num_points=100)
            sm_list.append(sm)
            times_list.append(step)

            fig.suptitle(f"Mapping Modes: {mapping}"
                         f"\n $t_0$ = {step}", fontsize=16) 

            ax[0,0].title.set_text('QNM mapping (real)')
            ax[0,0].pcolormesh(theta, phi, np.real(F), cmap=plt.cm.jet)

            ax[0,1].title.set_text('QNM expected (real)')
            ax[0,1].pcolormesh(theta, phi, np.real(G), cmap=plt.cm.jet)

            ax[1,0].title.set_text('QNM mapping (imag)')
            ax[1,0].pcolormesh(theta, phi, np.imag(F), cmap=plt.cm.jet)

            ax[1,1].title.set_text('QNM expected (imag)')
            ax[1,1].pcolormesh(theta, phi, np.imag(G), cmap=plt.cm.jet)

            ax[2,0].plot(times_list, sm_list, color='black')
            ax[2,0].set_ylabel('Spatial Mismatch')
            ax[2,0].set_xlabel('Time')

            pbar.update(1)
            if step == tstop - tstep:
                pbar.close()
            
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tstop, tstep), interval=100)

        return ani


    def animate_mapping_qnm_lm(self, sim, tstart=-100, tstop=200, tstep = 1, 
                               mapping_QNMs=[(2,2,0,1)], mapping=[(2,2,0,1)], l_max=9,
                               data_spherical_modes=None, mapping_spherical_modes=None, 
                               model_modes=None, projection='mollweide', t0=0):

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
        ax[0,0].axis('off')  
        ax[0,1].axis('off')  
        ax[1,0].axis('off')  
        ax[1,1].axis('off')  
        ax[2,0].axis('off')  
        ax[2,1].axis('off')  
        ax[0,0] = plt.subplot(3, 2, 1, projection=projection)
        ax[0,1] = plt.subplot(3, 2, 2, projection=projection)
        ax[1,0] = plt.subplot(3, 2, 3, projection=projection)
        ax[1,1] = plt.subplot(3, 2, 4, projection=projection)
        ax[2,0] = plt.subplot(3, 2, 5, projection=projection)
        ax[2,1] = plt.subplot(3, 2, 6, projection=projection)

        theta, phi = self.theta, self.phi

        if data_spherical_modes is None:
            data_spherical_modes = sim.h.keys()
        
        if mapping_spherical_modes is None:
            mapping_spherical_modes = [(l,m) for l in np.arange(2, l_max+1)
                                                            for m in np.arange(-l,l+1)]
            
        model_modes_label = model_modes 
        if model_modes is None:
            model_modes = [(lam,mu,n,p) for lam in np.arange(2, l_max+1)
                        for mu in np.arange(-lam, lam+1)
                           for n in np.arange(0, 3+1)
                              for p in (-1, +1)]
            model_modes_label = 'All'

        times = sim.times
        data_dict = sim.h

        pbar = tqdm(total=len(range(tstart, tstop, tstep)))

        def update(step):

            # Determine the complete GW spatial pattern (sum over all lm spherical modes)

            t0 = min(times, key=lambda x: abs(x - step))
            data_mask = (times == t0) 

            gw_map = sum(data_dict[lm][data_mask] * self.sYlm[lm[0], lm[1]] for lm in data_spherical_modes)
            gw_map /= np.max(np.abs(gw_map))

            # Determine the best fit QNM spatial pattern (sum over all lmnp QNM modes)

            best_fit = qnmfits.multimode_ringdown_fit(times, 
                                            data_dict, 
                                            modes=model_modes,
                                            Mf=sim.Mf,
                                            chif=sim.chif_mag,
                                            t0=step)
            
            # TODO: Check this is correct. How do you obtain the QNM mode reconstruction from the best fit?

            qnm_map = sum(sum(best_fit['weighted_C'][lm]) * self.sYlm[lm[0], lm[1]] for lm in data_spherical_modes)
            qnm_map /= np.max(np.abs(qnm_map))

            # Determine the best fit mapping of QNM modes

            best_fit = qnmfits.mapping_multimode_ringdown_fit(times, 
                                            data_dict, 
                                            modes=mapping_QNMs,
                                            Mf=sim.Mf,
                                            chif=sim.chif_mag,
                                            t0=step,
                                            mapping_modes=mapping,
                                            spherical_modes=mapping_spherical_modes)

            F = mode_mapping(np.pi/2-phi, theta, best_fit, mapping[0], l_max)

            fig.suptitle(f"QNM Model: {model_modes_label}, Mapping Modes: {mapping}"
                         f"\n $t_0$ = {step}", fontsize=16) 

            ax[0,0].title.set_text('GW data (real)')
            ax[0,0].pcolormesh(theta, phi, np.real(gw_map), cmap=plt.cm.jet)

            ax[0,1].title.set_text('GW data (imag)')
            ax[0,1].pcolormesh(theta, phi, np.imag(gw_map), cmap=plt.cm.jet)

            ax[1,0].title.set_text('QNM Model (real)')
            ax[1,0].pcolormesh(theta, phi, np.real(qnm_map), cmap=plt.cm.jet)

            ax[1,1].title.set_text('QNM Model (imag)')
            ax[1,1].pcolormesh(theta, phi, np.imag(qnm_map), cmap=plt.cm.jet)

            ax[2,0].title.set_text('QNM mapping (real)')
            ax[2,0].pcolormesh(theta, phi, np.real(F), cmap=plt.cm.jet)

            ax[2,1].title.set_text('QNM mapping (imag)')
            ax[2,1].pcolormesh(theta, phi, np.imag(F), cmap=plt.cm.jet)

            pbar.update(1)
            if step == tstop - tstep:
                pbar.close()
            
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tstop, tstep), interval=100)

        return ani