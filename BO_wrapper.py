import sys
sys.path.insert(1, '/home/chenyu/Desktop/git/STEMalign/')
from Bayesian_optimization import BOinterface

# set up BO hyper-parameters
abr_activate = [True, False, True, False, False, False, False, False, False, False, False, False]   # activated aberration coefficients
CNNpath = '/home/chenyu/Desktop/git/STEMalign/TorchModels/Test18_Nion_2ndorder_45mradApt_50mradLimit_emit+defocus_Adam_attempt02.pt'
filename = 'FirstOrder_45mrad'
aperture = 0
option_standardize = False
exposure_t = 500
remove_buffer = 1
n_init = 10
n_optimize = 50
apt_option = 40

Nion_BO = BOinterface(abr_activate, option_standardize = option_standardize, aperture = aperture, CNNpath = CNNpath, 
filename = filename, exposure_t= exposure_t, remove_buffer= remove_buffer, apt_option= apt_option)
Nion_BO.initialize_GP(n_init)
Nion_BO.run_optimization(n_optimize)
Nion_BO.saveresults()