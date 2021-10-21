import sys
import numpy as np
import threading
import matplotlib.pyplot as plt
from os.path import exists

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

'''
TODO: Read the default parameter before running optimization.
TODO: Add option to set to default or set to best seen parameter.
'''
class BOinterface():
    '''
    Main function that set up the input parameters and run the Bayesian optimization.
    '''
    def __init__(self, abr_activate, option_standardize, aperture, CNNpath, filename):

        # setup basic parameters
        self.ndim = sum(abr_activate)            # number of variable parameters
        self.aperture = aperture            # option to include cutoff aperture, in mrad
        self.niter = 50                     # number of iterations after initialization
        self.dtype = torch.double
        self.option_standardize = option_standardize
        self.CNNoption = 1
        # self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cuda:1')   # possible command to select from multiple GPUs
        self.device = "cpu"                 # hard coded to cpu for now, need to find a way to move all the model weights to the desired device
        CNNpath = CNNpath 

        # initialize the interface that talks to Nion swift.
        self.Nion = Nion_interface(act_list = abr_activate, readDefault = True, detectCenter = True, exposure_t = 500, remove_buffer = False)
        self.default = self.Nion.default
        
        # initialize the CNN model used to run predictions.
        self.model = self.loadCNNmodel(CNNpath)
        self.model.eval()

        # initialize the lists to save the results
        self.best_observed_value = []
        self.best_seen_ronchigram = np.zeros([128, 128])
        self.best_par = np.zeros(self.ndim)
        self.ronchigram_list = []

        # readin the name for saving the results
        self.filename = filename

    '''
    Function to load CNN model from path.
    Input: path to the torch model.
    '''
    def loadCNNmodel(self, path):
        model = Net(device = self.device, linear_shape = 256).to(self.device)
        state_dict = torch.load(path, map_location = self.device)
        model.load_state_dict(state_dict)
        return model

    '''
    Function to set objective based on CNN prediction.
    Input: 128x128 numpy array as the input to CNN.
    TODO: add scale_range, scale_range_aperture, and aperture_generator function here.
    TODO: change the limit in the aperture generator from 50 to a variable
    '''
    def getCNNprediction(self):
        acquire_thread = threading.Thread(target = self.Nion.acquire_frame())
        acquire_thread.start()
        frame_array = self.Nion.frame

        frame_array = self.Nion.scale_range_aperture(frame_array, 0, 1)
        if self.aperture != 0:
            frame_array = frame_array * self.Nion.aperture_generator(128, 50, self.aperture)
        new_channel = np.zeros(frame_array.shape)
        img_stack = np.dstack((frame_array, new_channel, new_channel))
        x = torch.tensor(np.transpose(img_stack)).to(self.device)
        x = x.unsqueeze(0).float()
        prediction = self.model(x)
        return frame_array, 1 - prediction[0][0].cpu().detach().numpy()

    '''
    # Function that initialize the GP and MLL model, with n random starting points.
    # Input:
    # n: int, number of datapoints to gerate
    '''

    def initialize_GP(self, n):

        # generate random initial training data
        self.train_X = torch.rand(n, self.ndim, device = self.device, dtype = self.dtype)
        output_y = []
        best_y = 0

        for i in range(self.train_X.shape[0]):
            self.Nion.setX(np.array(self.train_X[i,:]))
            pred = self.getCNNprediction()
            if pred[1] > best_y:
                self.best_seen_ronchigram = pred[0]
                best_y = pred[1]
            output_y.append(pred[1])
        self.train_Y = torch.tensor(output_y).unsqueeze(-1)

        if self.option_standardize:
            self.outcome_transformer = Standardize( m = 1,
            batch_shape = torch.Size([]),
            min_stdv = 1e-08)
            self.gp = SingleTaskGP(self.train_X, self.train_Y, outcome_transform = self.outcome_transformer)
        else:
            self.gp = SingleTaskGP(self.train_X, self.train_Y)
            
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.bounds = torch.stack([torch.zeros(self.ndim, device = self.device), torch.ones(self.ndim, device = self.device)])
        self.best_observed_value.append(best_y)
        self.ronchigram_list.append(self.best_seen_ronchigram)
        return

    '''
    Function to run one iteration on the Bayesian optimization and update both gp and mll.
    '''
    def run_iteration(self):
        fit_gpytorch_model(self.mll)
        UCB = UpperConfidenceBound(self.gp, beta = 2)    
        candidate, acq_value = optimize_acqf(
            UCB, bounds=self.bounds, q = 1, num_restarts=5, raw_samples=20,
        )
        new_x = candidate.detach()
        print(new_x)
        
        self.Nion.setX(np.array(new_x[0]))
        result = self.getCNNprediction()
        new_y = torch.tensor(result[1]).unsqueeze(-1).unsqueeze(-1)
        self.train_X = torch.cat([self.train_X, new_x])
        self.train_Y = torch.cat([self.train_Y, new_y])

        if result[1] > self.best_observed_value[-1]:
            self.best_par = np.array(new_x[0])
            self.best_value = result[1]
            self.best_seen_ronchigram = result[0]
            self.best_observed_value.append(result[1])
        else:
            self.best_observed_value.append(self.best_observed_value[-1])
        self.ronchigram_list.append(result[0])

        # update GP model using dataset with new datapoint
        if self.option_standardize:
            self.gp = SingleTaskGP(self.train_X, self.train_Y, outcome_transform = self.outcome_transformer)
        else:
            self.gp = SingleTaskGP(self.train_X, self.train_Y)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

    '''
    Function to run the full Bayesian optimization for niter iterations.
    Input:
    niter: int, number of iterations to run
    '''

    def run_optimization(self, niter):
        for i in range(niter):
            self.run_iteration()
            print(f"Iteraton number {i}, current best seen value {self.best_observed_value[-1]}")
        return

    '''
    Function ot save the process and results of Bayesian optimization.
    TODO: Try to find a way to save the whole GP model.
    '''
    def saveresults(self):
        train_X = self.train_X.cpu().detach().numpy()
        train_Y = self.train_Y.cpu().detach().numpy()
        index = 0
        temp = self.filename + 'X_' + str(index) + '.npy'
        if exists(temp):
            index += 1
            temp = self.filename + 'X_' + str(index) + '.npy'

        np.save(self.filename + str(index) + 'X_' + '.npy', train_X)
        np.save(self.filename + str(index) + 'Y_' + '.npy', train_Y)
        np.save(self.filename + str(index) + 'Ronchigram_' + '.npy', np.array(self.ronchigram_list))
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