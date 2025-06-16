'''
Task 2.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
#from matplotlib.colormaps import get_cmap
from matplotlib.colors import Normalize
from threading import Thread
import multiprocessing
import data, ic, solver

filepath = '/home/jaim/flds.tot.00410'
# System params:
n = int(1e5) # number of particles
N = int(1e3) # number of time steps, ~5e3 for equilibration
# Plot params:
N_plot = 64 # number of time steps to plot
N_bins = 96 # number of histogram bins
# Tristan-v2 numerical params:
cc = .2 # CFL condition: cc <= 1
c_omp = 10
# Tristan-v2 plasma physics params:
sigma = 10
beta_rec = .1
gammas_syn = [np.inf, 10, 6, 3]
gammas_ic = [np.inf, 15, 8, 4]
cool_lim = 1

n_threads = multiprocessing.cpu_count()
d = data.read(filepath)
dx, gs = data.preprocess(d, sigma) # cell size, grid size
dt = dx # time step
Dt = dt*N # duration
B_norm = cc**2*sigma**.5/c_omp
e_mean = np.mean((d['ex']**2+d['ey']**2+d['ez']**2)**.5)
b_mean = np.mean((d['bx']**2+d['by']**2+d['bz']**2)**.5)
print(f'{n_threads=}')
print(f'{dx=:.2e}')
print(f'{dt=:.2e}')
print(f'{Dt=:.2e}')
print(f'{B_norm=:.2e}')
print(f'{e_mean=:.2e}')
if e_mean > 1e1 or e_mean < 1e-1: print(f'WARNING: extreme e_mean')
print(f'{b_mean=:.2e}')
if b_mean > 1e1 or b_mean < 1e-1: print(f'WARNING: extreme b_mean')

pos_0, vel_0 = ic.generate_1(n, dx, gs, 1)

plt.rcParams['font.family'] = 'CMU' # latex font, ok if error
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (9, 6)
cmap = get_cmap('inferno')

# Integration and plotting the Maxwell–Juttner typa spectra.
e = [d['ex'], d['ey'], d['ez']]
b = [d['bx'], d['by'], d['bz']]
m = -n//(-n_threads)
plt.figure(1)
for k, (gamma_syn, gamma_ic) in enumerate(zip(gammas_syn, gammas_ic)):
  pos, vel = np.copy(pos_0), np.copy(vel_0)
  params_basic = (cc, B_norm, dx, gs)
  params_cool = (beta_rec, gamma_syn, gamma_ic, cool_lim)
  plt.subplot(2, 2, k+1)
  plt.xscale('log'); plt.yscale('log')
  for i in range(N):
    #print(f'{i+1:{len(str(N))}}/{N}')
    threads = [Thread(target=solver.step,
      args=(pos[:,m*j:m*(j+1)], vel[:,m*j:m*(j+1)], e, b, *params_basic, *params_cool))
      for j in range(n_threads)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if i % (N//N_plot) == 0:
      gamma = np.sqrt(1+vel[0]**2+vel[1]**2+vel[2]**2)
      hist, bin_edges = np.histogram(np.log(gamma-1), density=True, bins=N_bins)
      bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
      plt.plot(np.exp(bin_centers), np.exp(bin_centers)*hist, color=cmap(i/N))
  (scalar_mappable := ScalarMappable(cmap=cmap, norm=Normalize(0, (N-1)*dt*cc/(dx*gs)))).set_array([])
  cbar = plt.colorbar(scalar_mappable,ax=plt.gca(),label='$tc/L$') 
  plt.xlim(5e-3, 1e2); plt.ylim(1e-5, 2e1)
  plt.xlabel('$\gamma-1$'); plt.ylabel('$n(\gamma-1)$')
  if gamma_syn == np.inf: gamma_syn = '\infty'
  if gamma_ic == np.inf: gamma_ic = '\infty'
  plt.title(f'$\gamma_\\mathrm{{syn}}={gamma_syn}$ $\gamma_\\mathrm{{iC}}={gamma_ic}$' if gamma_syn != '\infty' or gamma_ic != '\infty' else 'no radiative cooling')
plt.tight_layout()
plt.savefig('3.pdf', bbox_inches='tight')
plt.show()
