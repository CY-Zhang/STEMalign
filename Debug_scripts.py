import sys
import numpy as np
import threading
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/chenyu/Desktop/git/STEMalign/')
from Nion_interface import Nion_interface
from TorchCNN import Net

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize

abr_activate = [True, False, True, False, False, False, False, False, False, False, False, False]   # activated aberration coefficients
ndim = sum(abr_activate)            # number of variable parameters
initial_count = 10                  # number of initial probe data points
option_standardize = True           # option to standardize training data for GP
aperture = 40                  # option to include cutoff aperture, in mrad
niter = 50                     # number of iterations after initialization
device = ("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# device = torch.device('cuda:1')   # possible command to select from multiple GPUs
CNNpath = '/home/chenyu/Desktop/git/STEMalign/TorchModels/Test6_Normalize_HighCs_emitxdefocus_Adam_attempt02.pt'


# Debug the CNN loading part
model = Net(linear_shape = 256, device = device)
state_dict = torch.load(CNNpath, map_location = device)
model.load_state_dict(state_dict)
model.eval()

# Debug single frame acquisition from Nion
Nion = Nion_interface(act_list = abr_activate, readDefault = True, detectCenter = True, exposure_t = 500, remove_buffer = False)
acquire_thread = threading.Thread(target = Nion.acquire_frame())
acquire_thread.start()
frame_array = Nion.frame

# Debug CNN prediction
frame_array = Nion.scale_range(frame_array, 0, 1)
if aperture != 0:
    frame_array = frame_array * Nion.aperture_generator(128, 50, aperture)
new_channel = np.zeros(frame_array.shape)
img_stack = np.dstack((frame_array, new_channel, new_channel))
x = torch.tensor(np.transpose(img_stack)).to(device)
x = x.unsqueeze(0).float()
prediction = model(x)

# Debug the data initialization part
# generate random initial training data
train_x = torch.rand(initial_count, ndim, device = device, dtype = torch.double)
output_y = []
for i in range(train_x.shape[0]):
    Nion.setX(np.array(train_x[i,:]))
    Nion.acquire_frame()
    frame_array = Nion.frame
    frame_array = Nion.scale_range(frame_array, 0, 1)
    if aperture != 0:
        frame_array = frame_array * Nion.aperture_generator(128, 50, aperture)
    new_channel = np.zeros(frame_array.shape)
    img_stack = np.dstack((frame_array, new_channel, new_channel))
    x = torch.tensor(np.transpose(img_stack)).to(device)
    x = x.unsqueeze(0).float()
    output_y.append(model(x))
train_y = torch.tensor(output_y).unsqueeze(-1)

# Debug the BO iteration part
gp = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
bounds = torch.stack([torch.zeros(ndim, device = device), torch.ones(ndim, device = device)])
fit_gpytorch_model(mll)

UCB = UpperConfidenceBound(gp, beta = 2)    
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q = 1, num_restarts=5, raw_samples=20,
)
new_x = candidate.detach()
print(new_x)

Nion.setX(np.array(new_x[0]))
result = getCNNprediction()
new_y = torch.tensor(result[1]).unsqueeze(-1).unsqueeze(-1)
train_X = torch.cat([train_X, new_x])
train_Y = torch.cat([train_Y, new_y])