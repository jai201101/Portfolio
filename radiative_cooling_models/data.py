import numpy as np
import h5py

def read(filepath):
  with h5py.File(filepath, 'r') as fp:
    return {k:np.ascontiguousarray(np.array(fp[k]).T) for k in fp.keys()} # (z,y,x) -> (x,y,z)

def preprocess(data, sigma):
  # Get the cell size. (Assuming cubic.)
  dx = data['xx'][1,0,0]-data['xx'][0,0,0]
  # Get the grid size. (Assuming cubic.)
  gs = data['xx'].shape[0]
  # Rescaling.
  for k in ['bx', 'by', 'bz', 'ex', 'ey', 'ez']: data[k] /= sigma/2/dx**3
  return dx, gs
