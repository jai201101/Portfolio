'''
Task 5.
'''

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from threading import Thread
import multiprocessing
import data, ic, solver

filepath = '/home/jaim/flds.tot.00410'
# System params:
n = int(4e5) # number of particles
N = int(5e3) # number of time steps, ~5e3 for equilibration
# Plot params:
N_bins = 32 # number of histogram bins
bins = (np.linspace(-1.5, 3.5, N_bins+1), np.linspace(0, np.pi, N_bins+1)) # histogram bins
# Tristan-v2 numerical params:
cc = .2 # CFL condition: cc <= 1
c_omp = 10
# Tristan-v2 plasma physics params:
sigma = 10
beta_rec = .1
gammas_syn = [20, 8, 8, np.inf, 4, np.inf]
gammas_ic = [20, 8, np.inf, 8, np.inf, 4]
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
plt.rcParams['figure.figsize'] = (9, 8)
cmap = get_cmap('inferno')

# Integration and plotting.
e = [d['ex'], d['ey'], d['ez']]
b = [d['bx'], d['by'], d['bz']]
m = -n//(-n_threads)
for k, (gamma_syn, gamma_ic) in enumerate(zip(gammas_syn, gammas_ic)):
  pos, vel = np.copy(pos_0), np.copy(vel_0)
  params_basic = (cc, B_norm, dx, gs)
  params_cool = (beta_rec, gamma_syn, gamma_ic, cool_lim)
  for i in range(N):
    print(f'{i+1:{len(str(N))}}/{N}')
    threads = [Thread(target=solver.step,
      args=(pos[:,m*j:m*(j+1)], vel[:,m*j:m*(j+1)], e, b, *params_basic, *params_cool))
      for j in range(n_threads)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
  # Calculate histograms, 2D and 2D->1D contracted.
  b_int = np.array([
    map_coordinates(b[0], pos/dx, order=1, mode='grid-wrap'),
    map_coordinates(b[1], pos/dx, order=1, mode='grid-wrap'),
    map_coordinates(b[2], pos/dx, order=1, mode='grid-wrap')])*B_norm
  gamma = np.sqrt(1+vel[0]**2+vel[1]**2+vel[2]**2)
  alpha = np.arccos(np.sum(vel*b_int, axis=0)/(np.linalg.norm(vel, axis=0)*np.linalg.norm(b_int, axis=0)))
  hist, bin_edges_gamma, bin_edges_alpha = np.histogram2d(np.log(gamma-1), alpha, density=True, bins=bins)
  bin_centers_gamma = (bin_edges_gamma[:-1]+bin_edges_gamma[1:])/2
  bin_centers_alpha = (bin_edges_alpha[:-1]+bin_edges_alpha[1:])/2
  norm_alpha = np.sum(hist, axis=1)
  good = np.where(norm_alpha)
  hist_alpha = np.sum(hist*bin_centers_alpha, axis=1)[good]/norm_alpha[good]
  # Plotting 2D histograms.
  if gamma_syn == np.inf: gamma_syn = '\infty'
  if gamma_ic == np.inf: gamma_ic = '\infty'
  plt.figure(1)
  plt.subplot(-len(gammas_syn)//(-2), 2, k+1)
  plt.imshow(hist.T, origin='lower', extent=(bin_edges_gamma[0], bin_edges_gamma[-1], bin_edges_alpha[0], bin_edges_alpha[-1]), interpolation='none', cmap=cmap)
  (scalar_mappable := ScalarMappable(cmap=cmap, norm=Normalize(0, np.max(hist)))).set_array([])
  cbar = plt.colorbar(scalar_mappable, label='$n(\ln(\gamma-1),\\alpha)$')
  plt.xlabel('$\ln(\gamma-1)$'); plt.ylabel('$\\alpha$')
  plt.yticks([0, np.pi/2, np.pi], ['0', '$\\pi/2$', '$\\pi$'])
  plt.title(f'$\gamma_\\mathrm{{syn}}={gamma_syn}$ $\gamma_\\mathrm{{iC}}={gamma_ic}$' if gamma_syn != '\infty' or gamma_ic != '\infty' else 'no radiative cooling')
  # Plotting 1D histograms.
  plt.figure(2)
  plt.subplot(-len(gammas_syn)//(-2), 2, k+1)
  plt.plot(bin_centers_gamma[good], hist_alpha, c='black')
  plt.xlabel('$\ln(\gamma-1)$'); plt.ylabel('$\\alpha$')
  plt.yticks([0, np.pi/2, np.pi], ['0', '$\\pi/2$', '$\\pi$'])
  plt.xlim(bin_centers_gamma[good][0], bin_centers_gamma[good][-1])
  plt.title(f'$\gamma_\\mathrm{{syn}}={gamma_syn}$ $\gamma_\\mathrm{{iC}}={gamma_ic}$' if gamma_syn != '\infty' or gamma_ic != '\infty' else 'no radiative cooling')
plt.figure(1)
plt.tight_layout()
plt.savefig('4_1.pdf', bbox_inches='tight')
plt.figure(2)
plt.tight_layout()
plt.savefig('4_2.pdf', bbox_inches='tight')
