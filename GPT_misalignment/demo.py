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

xlim, ylim, shadow = sim(                
                 S1CV  = 0.0,
                 S1CH  = 0.0,
                 S2CV  = 0.0,
                 S2CH  = 0.0,
                 S3CV  = 0.0,
                 S3CH  = 0.0,
                 H1CV  = 0.0,
                 H1CH  = 0.0,
                 S4CV  = 0.0,
                 S4CH  = 0.0,
                 S5CV  = 0.0,
                 S5CH  = 0.0,
                 H2CV  = 0.0,
                 H2CH  = 0.0,
                 S6CV  = 0.0,
                 S6CH  = 0.0,
                 S7CV  = 0.0,
                 S7CH  = 0.0,
                 ObjCH = 0.0,
                 ObjCV = 0.0,
                 H1 = 1498.216,
                 H2 = 1498.694,
                 S1 = 2.5e5,
                 S2 = 2.5e5,
                 S3 = 119931.5,
                 S4 = 648691.415,
                 S6 = 390000,
                 S7 =-654100,  #-654100.0, controls the defocus
                 alpha = alpha*5,
                 Obj=-3.7503e6-220,
                 erL = 0,
                 erTh = 0,)

plt.imshow((shadow - np.amin(shadow))/(np.amax(shadow) - np.amin(shadow)), extent=[-xlim*L, xlim*L, -ylim*L, ylim*L], cmap='gray')
fname="shadow{}.png".format(H1)
plt.xlabel(r"$x^\prime \ \mathrm{(mrad)}$")
plt.ylabel(r"$y^\prime \ \mathrm{(mrad)}$")
plt.savefig(fname)
plt.show()
