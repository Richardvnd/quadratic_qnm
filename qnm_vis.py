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


    def plot_spherical_modes(self, ax, times, data_dict, theta, phi, t0=0, spherical_modes=None):
        if spherical_modes is None:
            spherical_modes = list(data_dict.keys())

        t0 = min(times, key=lambda x: abs(x - t0))
        data_mask = (times == t0) 

        map = np.zeros_like(self.theta, dtype=complex)
        map = sum(data_dict[lm][data_mask] * self.sYlm[lm[0], lm[1]] for lm in spherical_modes)
        map /= np.max(np.abs(map))

        ax[0].title.set_text('Real')
        ax[0].pcolormesh(theta, phi, np.real(map), cmap=plt.cm.viridis)

        ax[1].title.set_text('Imaginary')
        ax[1].pcolormesh(theta, phi, np.imag(map), cmap=plt.cm.viridis)

        return ax
    

    def animate_spherical_modes(self, times, data_dict, spherical_modes=None, 
                                tstart = 10000, tstop = 11000, tstep = 50, projection='mollweide'): 

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
        ax[0].axis('off')  
        ax[1].axis('off')  
        ax[0] = plt.subplot(1, 2, 1, projection=projection)
        ax[1] = plt.subplot(1, 2, 2, projection=projection)

        theta, phi = self.theta, self.phi

        def update(step):
            self.plot_spherical_modes(ax, times, data_dict, theta, phi, step, spherical_modes)
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tstop, tstep), interval=50)

        return ani
    

    def animate_qnms(self, sim, tstart=-100, tend=200, modes=[(2,2,0,1)], tstep = 1, projection='mollweide'):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
        ax[0].axis('off')  
        ax[1].axis('off')  
        ax[0] = plt.subplot(1, 2, 1, projection=projection)
        ax[1] = plt.subplot(1, 2, 2, projection=projection)

        theta, phi = self.theta, self.phi

        spherical_modes = sim.h.keys()

        def update(step):
            best_fit = qnmfits.multimode_ringdown_fit(sim.times, 
                                            sim.h, 
                                            modes=modes,
                                            Mf=sim.Mf,
                                            chif=sim.chif_mag,
                                            t0=step)
            
            # TODO: Check this is correct. How do you obtain the QNM mode reconstruction from the best fit?

            map = np.zeros_like(self.theta, dtype=complex)
            map = sum(sum(best_fit['weighted_C'][lm]) * self.sYlm[lm[0], lm[1]] for lm in spherical_modes)
            map /= np.max(np.abs(map))

            ax[0].title.set_text('Real')
            ax[0].pcolormesh(theta, phi, np.real(map), cmap=plt.cm.viridis)

            ax[1].title.set_text('Imaginary')
            ax[1].pcolormesh(theta, phi, np.imag(map), cmap=plt.cm.viridis)
            
            return ax 

        ani = FuncAnimation(fig, update, frames=range(tstart, tend, tstep), interval=50)

        return ani

