import sys
import numpy as np
import threading
import matplotlib.pyplot as plt

sys.path.insert(1, 'C:/Users/ASUser/Downloads/Bayesian-optimization-using-Gaussian-Process/')
from Nion_interface import Nion_interface
from TorchCNN import Net

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize

'''
Main function that set up the input parameters and run the Bayesian optimization.
'''
def main(self):
    # setup basic parameters
    abr_activate = [True, False, True, False, False, False, False, False, False, False, False, False]   # activated aberration coefficients
    ndim = sum(abr_activate)            # number of variable parameters
    initial_count = 10                  # number of initial probe data points
    option_standardize = True           # option to standardize training data for GP
    self.aperture = 40                  # option to include cutoff aperture, in mrad
    self.niter = 50                     # number of iterations after initialization
    self.device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:1')   # possible command to select from multiple GPUs
    CNNpath = 'C:/Users/ASUser/Downloads/Bayesian-optimization-using-Gaussian-Process/CNNmodels/VGG16_nion_2ndOrder_45mradEmit+defocus_45mradApt.h5'
    

    # initialize the interface that talks to Nion swift.
    self.Nion = Nion_interface(act_list = abr_activate, readDefault = True, detectCenter = True, exposure_t = 500, remove_buffer = False)
    # initialize the CNN model used to run predictions.
    self.model = self.loadCNNmodel(CNNpath, device = self.device)

    if option_standardize:
        outcome_transformer = Standardize( m = 1,
        batch_shape = torch.Size([]),
        min_stdv = 1e-08)
        
    train_X, train_Y = generate_initial_data(n = initial_count, ndim = ndim)

    if option_standardize:
        gp = SingleTaskGP(train_X, train_Y, outcome_transform = outcome_transformer)
    else:
        gp = SingleTaskGP(train_X, train_Y)
        
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    bounds = torch.stack([torch.zeros(self.ndim, device = self.device), torch.ones(6, device = self.device)])

    best_observed_value = []
    best_seen_ronchigram = np.zeros([128, 128])

    for iteration in range(self.niter):

        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(gp, beta = 2)    
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q = 1, num_restarts=5, raw_samples=20,
        )
        new_x = candidate.detach()
        print(new_x)
        
        self.setX(self, np.array(new_x[0]))
        result = self.getCNNprediction()
        new_y = torch.tensor(result[1]).unsqueeze(-1).unsqueeze(-1)
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])

        if not best_observed_value:
            best_par = np.array(new_x[0])
            best_value = np.array(new_y[0][0])
            best_seen_ronchigram = result[0]

        elif result[1] > best_value:
            best_par = np.array(new_x[0])
            best_value = result[1]
            best_seen_ronchigram = result[0]

        best_observed_value.append(best_value)

        # update GP model using dataset with new datapoint
        if option_standardize:
            gp = SingleTaskGP(train_X, train_Y, outcome_transform = outcome_transformer)
        else:
            gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        print(iteration, best_value)
'''
Function ot save the process and results of Bayesian optimization.
TODO: Try to find a way to save the whole GP model.
'''
def saveresults(self, best_observed_value):
    return


'''
Function to plot the Bayesian optimization results.
TODO: Possibly add the model's prediction of a single dimension.
Input: 
best_observed_value: numpy array saving the best observed value for each iteration.
best_seen_ronchigran: 2D numpy array saving the ronchigram that correspond to the optimized parameter.

'''
def plotresults(self, best_observed_value, best_seen_ronchigram):
    plt.plot(best_observed_value)
    plt.imshow(best_seen_ronchigram)
    plt.show()
    return

'''
Function to load CNN model from path.
Input: path to the torch model.
'''
def loadCNNmodel(self, path):
    model = Net(linear_shape = 256).to(self.device)
    pretrained_dict = torch.load(path)
    model.load_pretrained(pretrained_dict)
    return model

'''
Function to set objective based on CNN prediction.
Input: 128x128 numpy array as the input to CNN.
'''
def getCNNprdiction(self):
    acquire_thread = threading.Thread(target = self.Nion.acquire_frame())
    acquire_thread.start()
    frame_array = self.Nion.frame
    
    if self.CNNoption == 1:
        if 'self.model' not in locals():
            print("CNN model not loaded.")
            return frame_array, 1
        frame_array = self.scale_range_aperture(frame_array, 0, 1)
        if self.aperture != 0:
            frame_array = frame_array * self.aperture_generator(128, 50, self.aperture)
        new_channel = np.zeros(frame_array.shape)
        img_stack = np.dstack((frame_array, new_channel, new_channel))
        x = torch.tensor(np.transpose(img_stack)).to(self.device)
        x = x.unsqueeze(0).float()
        prediction = self.model(x)
        return frame_array, 1 - prediction[0][0]
    else:
        print("Testing without CNN.")
        return frame_array, 1

'''
# Function that generate n random data point with noise.
# Input:
# n: number of datapoints to gerate
# ndim: number of random parameters
'''

def generate_initial_data(self, n, ndim):
    # generate random initial training data
    train_x = torch.rand(n, ndim, device = self.device, dtype = self.dtype)
    output_y = []
    for i in range(train_x.shape[0]):
        self.Nion.setX(self, np.array(train_x[i,:]))
        output_y.append(self.getCNNprediction())
    train_y = torch.tensor(output_y).unsqueeze(-1)
    return train_x, train_y

# Entry point to the main function
if __name__ == "__main__":
    main()

