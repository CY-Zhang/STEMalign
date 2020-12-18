import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

sampleL = 5e-10
sampleN = 200
pad     = 50
x_fine, y_fine = np.meshgrid(np.linspace(0, sampleL, sampleN),
                             np.linspace(0, sampleL, sampleN))
rough = np.random.binomial(1, 0.5, size=(sampleN, sampleN)).astype(float)
rough_pad = np.zeros((sampleN+pad, sampleN+pad))
rough_pad[25:225,25:225] = rough
rough_pad[25:225,0:25]   = rough[   :  ,-25:]
rough_pad[25:225,225:]   = rough[   :  ,   :25]
rough_pad[0:25, 25:225]  = rough[-25:  ,   :]
rough_pad[225:, 25:225]  = rough[   :25,   :]
rough_pad[0:25,0:25]     = rough[-25:  ,-25:]
rough_pad[0:25,225:]     = rough[-25:  ,   :25]
rough_pad[225:,225:]     = rough[   :25,   :25]
rough_pad[225:,0:25]     = rough[   :25,-25:]
blurred_pad = gaussian_filter(rough_pad, sigma=3)
blurred = blurred_pad[25:225, 25:225]

test = np.zeros((sampleN*3, sampleN*3))
for i in range(0,3):
    for j in range(0,3):
        test[i*200:(i+1)*200, j*200:(j+1)*200] = blurred

plt.imshow(test)
plt.show()

trnsmssn = interpolate.interp2d(x_fine.flatten(), y_fine.flatten(), blurred.flatten())

FILENAME = "trnsmssn.pickle"
with open(FILENAME, "wb") as f:
    pickle.dump(trnsmssn, f)
