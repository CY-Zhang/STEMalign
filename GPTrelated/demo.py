import pickle
from uscope import sim
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
#                                      the input to sim() is a list of  magnet strengths,
shadow, x_grid, y_grid, kx_grid, ky_grid = sim(
	S0    = 0.0,
    H1    = 0.2, #0.2, #-2.9,
    H2    = 0.2, #0.2, #-2.9,
	Obj   = 1.30306, #1.30305,
	alpha = 2.0e-5, #1.1e-5
	seed  = 0,
	erL   = 0.0,
    erTh  = 0.0)

k_abs = np.power(np.power(kx_grid, 2) + np.power(ky_grid, 2), 0.5)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=[12,6])
kmap = ax0.imshow(np.flip(k_abs, axis=0), extent=[x_grid[0,0], x_grid[-1,-1], y_grid[0,0], y_grid[-1,-1]])
ax0.set_xlabel("x (m)")
ax0.set_ylabel("y (m)")
cbar = fig.colorbar(kmap, ax=ax0)
cbar.set_label(r"$|k_\perp| \quad \mathrm{(m)}^{-1}$")
ax1.quiver(x_grid, y_grid, kx_grid, ky_grid)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
plt.show()
fig.savefig('abr_gradient.png')

fig = plt.figure()
plt.imshow(shadow)
plt.show()
plt.savefig('ronchigram.png')

