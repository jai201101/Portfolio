'''
Example/Test 1: initial conditions.
'''

import numpy as np
import matplotlib.pyplot as plt
import data, ic

filepath = '../../data/flds.tot.00410'
# System params:
n = int(1e5) # number of particles
# Tristan-v2 numerical params:
cc = .2 # CFL condition: cc <= 1
c_omp = 10
# Tristan-v2 plasma physics params:
sigma = 10

d = data.read(filepath)
dx, gs = data.preprocess(d, sigma)

# pos_i, vel, sampling_eff = ic.generate_2(d['dens1'], 1, n)
# print(f'{sampling_eff*100=:.2f}%')
pos, vel = ic.generate_1(n, dx, gs, 1)

slice = 42
i = np.where(np.logical_and(slice <= pos[0]/dx, pos[0]/dx < slice+1))
pos_slice, vel_slice = pos[:,i], vel[:,i]
plt.figure(1)
plt.imshow(d['dens1'][slice,:,:].T, cmap='inferno', origin='lower', extent=(0, dx*gs, 0, dx*gs), interpolation='none') # (y,z) -> (z,y), y horizontal
plt.plot(pos_slice[1], pos_slice[2], '.', c='white', ms=1)
# plt.quiver(pos_slice[1], pos_slice[2], vel_slice[1], vel_slice[2], color='white')
plt.title(f'x={slice*dx}')
plt.xlabel('y'); plt.ylabel('z')
plt.tight_layout()
plt.savefig('1.pdf', bbox_inches='tight')
plt.show()
