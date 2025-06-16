import numpy as np
from scipy.ndimage import map_coordinates

def step(pos, vel, e, b, cc, B_norm, dx, gs, beta_rec, gamma_syn, gamma_ic, cool_lim):
  pos_i = pos/dx
  vel_0 = np.copy(vel)
  # Interpolate electric and magnetic fields.
  e_0 = np.array([
    map_coordinates(e[0], pos_i, order=1, mode='grid-wrap'),
    map_coordinates(e[1], pos_i, order=1, mode='grid-wrap'),
    map_coordinates(e[2], pos_i, order=1, mode='grid-wrap')])
  b_0 = np.array([
    map_coordinates(b[0], pos_i, order=1, mode='grid-wrap'),
    map_coordinates(b[1], pos_i, order=1, mode='grid-wrap'),
    map_coordinates(b[2], pos_i, order=1, mode='grid-wrap')])
  # Boris mover, to get half step momentum estimate.
  e, b = np.copy(e_0), np.copy(b_0)
  c1 = -.5*B_norm
  e *= c1
  c1 /= cc
  b *= c1
  #  1st half electric acceleration.
  vel *= cc
  vel += e
  #  1st half magnetic rotation.
  b *= cc/np.sqrt(cc**2+vel[0]**2+vel[1]**2+vel[2]**2)
  c1 = 2/(1+b[0]**2+b[1]**2+b[2]**2)
  vel_B = np.array([
    (vel[0]+vel[1]*b[2]-vel[2]*b[1])*c1,
    (vel[1]+vel[2]*b[0]-vel[0]*b[2])*c1,
    (vel[2]+vel[0]*b[1]-vel[1]*b[0])*c1])
  #  2nd half magnetic rotation + 2nd half electric acceleration.
  vel[0] = (vel[0]+vel_B[1]*b[2]-vel_B[2]*b[1]+e[0])/cc
  vel[1] = (vel[1]+vel_B[2]*b[0]-vel_B[0]*b[2]+e[1])/cc
  vel[2] = (vel[2]+vel_B[0]*b[1]-vel_B[1]*b[0]+e[2])/cc
  # Apply synchrotron radiation, using the velocity estimate from Boris mover.
  if gamma_syn < np.inf:
    vel_mid = (vel+vel_0)/2
    gamma = np.sqrt(1+vel_mid[0]**2+vel_mid[1]**2+vel_mid[2]**2)
    e_bar = np.array([
      e_0[0]+(vel_mid[1]*b_0[2]-vel_mid[2]*b_0[1])/gamma,
      e_0[1]+(vel_mid[2]*b_0[0]-vel_mid[0]*b_0[2])/gamma,
      e_0[2]+(vel_mid[0]*b_0[1]-vel_mid[1]*b_0[0])/gamma])
    e_bar_sq = e_bar[0]**2+e_bar[1]**2+e_bar[2]**2
    beta_dot_e = (e[0]*vel_mid[0]+e[1]*vel_mid[1]+e[2]*vel_mid[2])/gamma
    chi_R_sq = np.abs(e_bar_sq-beta_dot_e**2)
    kappa_R = np.array([
      (b_0[2]*e_bar[1]-b_0[1]*e_bar[2])+(e_0[0]*beta_dot_e),
      (-b_0[2]*e_bar[0]+b_0[0]*e_bar[2])+(e_0[1]*beta_dot_e),
      (b_0[1]*e_bar[0]-b_0[0]*e_bar[1])+(e_0[2]*beta_dot_e)])
    c2 = B_norm*beta_rec/cc/gamma_syn**2
    if (c2*chi_R_sq*gamma > cool_lim).any():
      print(f'WARNING: synchrotron cooling limit exceeded, {np.max(c2*chi_R_sq*gamma):.2e} > {cool_lim:.2e}')
      c2 = cool_lim/(chi_R_sq*gamma)
    vel += c2*(kappa_R-chi_R_sq*gamma*vel_mid)
  # Apply inverse Compton radiation, using the newest velocity.
  if gamma_ic < np.inf:
    vel_mid = (vel+vel_0)/2
    gamma = np.sqrt(1+vel_mid[0]**2+vel_mid[1]**2+vel_mid[2]**2)
    c3 = B_norm*beta_rec/cc/gamma_ic**2
    if (c3*gamma > cool_lim).any():
      print(f'WARNING: inverse Compton cooling limit exceeded, {np.max(c3*gamma):.2e} > {cool_lim:.2e}')
      c3 = cool_lim/gamma
    vel -= c3*gamma*vel_mid
  # Update positions.
  pos += vel*cc
  # Enforce boundary conditions.
  for i in range(3):
    pos[i,np.where(pos[i] < 0)] += dx*gs
    pos[i,np.where(pos[i] >= dx*gs)] -= dx*gs
