import numpy as np
from scipy.ndimage import map_coordinates

'''
Uniform Maxwellian initial condition with distribution parameter a = sqrt(kT/m).
'''
def generate_1(n, dx, gs, a):
  pos = np.random.rand(3, n)*dx*gs
  vel = np.random.randn(3, n)*a
  return pos, vel

'''
Sample particles with a density proportional to density `rho` via a Monte Carlo
sampler, and Maxwellian velocities with parameter a = sqrt(kT/m).
'''
def generate_2(rho, a, n, batch_size=1024):
  rho_max = np.max(rho)
  pos_i = []
  i, j = 0, 0 # accepted, attempts
  while True:
    pos_i_cand = np.random.rand(3, batch_size)*160
    rho_cand = map_coordinates(rho, pos_i_cand, order=1, mode='grid-wrap')
    p = np.random.rand(batch_size) 
    pos_i += [pos_i_cand[:,(p < rho_cand/rho_max)]]
    i += pos_i[-1].shape[-1]
    j += batch_size
    if i >= n:
      pos_i = np.concatenate(pos_i, axis=1)[:,:n]
      vel = np.random.randn(3, n)*a
      return pos_i, vel, i/j
