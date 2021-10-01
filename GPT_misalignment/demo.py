import pickle
from uscope import sim
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

np.random.seed(seed=0)

S1 = 2.5e5
S2 = 2.5e5
S3 = 119931.5
S4 = 648691.415
H1 = 900.7
S6 = 390000
S7 =-654100  #-654100.0
alpha = 1.0e-4
Obj=-9.39e5
L = 1e3

# Default set of parameters
# xlim, ylim, shadow = sim(                
#                 H1 = 900.7,
#                 H2 = 900.7,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0,
#                 alpha = alpha*5,
#                 Obj=-9.390e5,)

# 4x Obj to get a smaller C3
# xlim, ylim, shadow = sim(                
#                 H1 = 1498.216,
#                 H2 = 1498.694,
#                 # H1 = 1960.01,
#                 # H2 = 1960.01-1.12,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0,
#                 alpha = alpha*5,
#                 Obj=-3.7503e6-220,)

# Check the uncorredcted state with 4x obj
# xlim, ylim, shadow = sim(                
#                 H1 = 0,
#                 H2 = 0,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0,
#                 alpha = alpha*7,
#                 Obj=-3.7503e6+50,) # this setup work well when crop rxy at 0.07 um, cz 02-20-2021

# Test different states found by the GP
xlim, ylim, shadow = sim(                
                H1 = 1.5045e+03,
                H2 = 1.5045e+03 -2.0965e+01,
                S1 = 2.5e5,
                S2 = 2.5e5,
                S3 = 1.1945e+05,
                S4 = 6.4994e+05,
                S6 = 3.9131e+05,
                S7 =-6.6482e+05,
                alpha = alpha*5,
                Obj=-3.7505e6,
                erL = 0,
                erTh = 1e-3,
                seed = 11)

plt.imshow(shadow, extent=[-xlim*L, xlim*L, -ylim*L, ylim*L])
fname="shadow{}.png".format(H1)
plt.xlabel(r"$x^\prime \ \mathrm{(mrad)}$")
plt.ylabel(r"$y^\prime \ \mathrm{(mrad)}$")
plt.savefig(fname)
plt.show()
