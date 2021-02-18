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
xlim, ylim, shadow = sim(                
                H1 = 940.7,
                H2 = 940.7,
                S1 = 2.5e5,
                S2 = 2.5e5,
                S3 = 119931.5,
                S4 = 648691.415,
                S6 = 390000,
                S7 =-654100,  #-654100.0, controls the defocus
                alpha = alpha*5,
                Obj=-9.3905e5,)

# Manually search for optimal S1-S7
# xlim, ylim, shadow = sim(                
# 		H1 = 890,
#                 H2 = 890,
#                 S1 = 2.0e5,
#                 S2 = 2.67e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 402000,
#                 S7 =-654100,  #-654100.0, controls the defocus
#                 alpha = alpha*5,
#                 Obj=-9.39e5,)

#Optimal Obj setup when both hexapoles are off
# xlim, ylim, shadow = sim(                
#                 H1 =  0,
#                 H2 = 0,
#                 S1 = 2.5e5,
#                 S2 = 2.5e5,
#                 S3 = 119931.5,
#                 S4 = 648691.415,
#                 S6 = 390000,
#                 S7 =-654100,  #-654100.0, controls the defocus
#                 alpha = alpha*5,
#                 Obj=-9.69e5 )  # -9.38e5 is the best one for Obj without H1, H2)

plt.imshow(shadow, extent=[-xlim*L, xlim*L, -ylim*L, ylim*L])
fname="shadow{}.png".format(H1)
plt.xlabel(r"$x^\prime \ \mathrm{(mrad)}$")
plt.ylabel(r"$y^\prime \ \mathrm{(mrad)}$")
plt.savefig(fname)
plt.show()
