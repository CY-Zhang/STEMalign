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

# Default set of parameters, S1-7 using the input from Cameron
# xlim, ylim, shadow = sim(                
#                 H1 = 900.7,
#                 H2 = 900.7,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0, controls the defocus
#                 alpha = alpha*5,
#                 Obj=-9.390e5,)

# Test for stronger Obj to eliminate the level of Cs
xlim, ylim, shadow = sim(                
                H1 = 1498.216,
                H2 = 1498.694,
                S1 = 2.5e5,
                S2 = 2.5e5,
                S3 = 119931.5,
                S4 = 648691.415,
                S6 = 390000,
                S7 =-654100,  #-654100.0, controls the defocus
                alpha = alpha*8,
                Obj=-3.7503e6-200,)
                # Obj=-9.390e5*4+5700,)

# Check the states with H1 and H2 off under strong obj
# xlim, ylim, shadow = sim(                
#                 H1 = 0,
#                 H2 = 0,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0, controls the defocus
#                 alpha = alpha*5,
#                 Obj=-3.7503e6-12500,)
#                 # Obj=-9.390e5*4+5700,)


#Optimal Obj setup when both hexapoles are off
# xlim, ylim, shadow = sim(                
#                 H1 = 0,
#                 H2 = 0,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0, controls the defocus
#                 alpha = alpha*5,
#                 Obj=-9.39e5 )  # -9.38e5 is the best one for Obj without H1, H2)

plt.imshow(shadow, extent=[-xlim*L, xlim*L, -ylim*L, ylim*L])
fname="shadow{}.png".format(H1)
plt.xlabel(r"$x^\prime \ \mathrm{(mrad)}$")
plt.ylabel(r"$y^\prime \ \mathrm{(mrad)}$")
plt.savefig(fname)
plt.show()
