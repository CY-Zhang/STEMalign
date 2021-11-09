import sys
sys.path.insert(1, 'C:/Users/ASUsers/Downloads/TorchBO/')
from Bayesian_optimization import BOinterface

# set up BO hyper-parameters
abr_activate = [False, False, False, True, True, False, False, False, False, False, False, False]   # activated aberration coefficients
CNNpath = 'C:/Users/ASUsers/Downloads/TorchModels/Test18_Nion_2ndorder_no1stOrder_45mradApt_50mradLimit_emit_Adam_attempt00.pt'
filename = 'C:/Users/ASUsers/Downloads/2ndorder_attempt00_45mrad_250ms_standardize_removebuffer_scale'
aperture = 0
option_standardize = False
exposure_t = 500
remove_buffer = 1
n_init = 10
n_optimize = 50
apt_option = 0

for _ in range(2):
    Nion_BO = BOinterface(abr_activate, option_standardize = option_standardize, aperture = aperture, CNNpath = CNNpath, 
    filename = filename, exposure_t= exposure_t, remove_buffer= remove_buffer, apt_option= apt_option)
    Nion_BO.initialize_GP(n_init)
    Nion_BO.run_optimization(n_optimize)
    Nion_BO.saveresults()