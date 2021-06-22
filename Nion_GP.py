from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1, '/home/chenyu/Desktop/GaussianProcess')
# GP related libaries
saveResultsQ = False
from modules.bayes_optimization import BayesOpt, negUCB, negExpImprove
from modules.OnlineGP import OGP
# Standard python libraries
import pickle
import numpy as np
import importlib
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

ndim = 12 #input dimension
acquisition_delay = 0

dev_ids =  [str(x + 1) for x in np.arange(ndim)] #creat device ids (just numbers)
# Iteration boundary for Nionswift-sim, in the order of C10, C12.x/y, C21.x/y, C23.x/y, C30, C32.x/y, C34.x/y
# Could be removed in real instrument
iter_bounds = [(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-7, 5e-7),(-5e-6, 5e-6),(-5e-6, 5e-6),(-3.5e-6, 3.5e-6),(-3.5e-6, 3.5e-6),(-5e-5, 5e-5),(-5e-5, 5e-5),(-3.5e-5, 3.5e-5),
(-3.5e-5, 3.5e-5),(-3.5e-5, 3.5e-5)] 
# randomize starting point, in the simulator, the global minimum is at zero point.
rs = np.random.RandomState()
start_point = [[x[0] + (rs.rand()*(x[1]-x[0])) for x in iter_bounds]]
print(start_point)

#creat machine interface
mi_module = importlib.import_module('machine_interfaces.machine_interface_Nion')
mi = mi_module.machine_interface(dev_ids = dev_ids, start_point = start_point, CNNoption = 1, CNNpath = '/home/chenyu/Desktop/GaussianProcess/CNNmodels/VGG16_test13_attempt06.h5') 

# Check the readout from machine interface
print(mi.x)
temp = mi.getState()
print(temp[1][0][0])