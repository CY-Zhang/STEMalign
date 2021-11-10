import sys
sys.path.insert(1, 'C:/Users/ASUsers/Downloads/TorchBO/')
from Bayesian_optimization import BOinterface
import time

# set up BO hyper-parameters
abr_activate = [True, True, True, True, True, False, False, False, False, False, False, False]   # activated aberration coefficients
# network to correct up to C23 by minimizing emittance.
CNNpath_1 = 'C:/Users/ASUsers/Downloads/TorchModels/Test18_Nion_2ndorder_no1stOrder_45mradApt_50mradLimit_emit_Adam_attempt00.pt' 
# network to correct first order by minimizing defocus.
CNNpath_2 = 'C:/Users/ASUsers/Downloads/TorchModels/Test18_Nion_defocus_45mradApt_50mradLimit_emit+defocus_Adam_attempt05.pt' 
filename = 'C:/Users/ASUsers/Downloads/2ndorder_attempt00_45mrad_250ms_standardize_removebuffer_scale'
aperture = (50, 0) # tuple for acquisition limit and aperture size
option_standardize = False
exposure_t = 250
remove_buffer = 1
n_init = 10
n_optimize = 50
apt_option = 0
acq_func_par = ('UCB', 2)

for _ in range(2):
    start = time.time()
    Nion_BO = BOinterface(abr_activate, option_standardize = option_standardize, aperture = aperture, CNNpath = CNNpath_1, 
    filename = filename, exposure_t= exposure_t, remove_buffer= remove_buffer, apt_option= apt_option, acq_func_par = acq_func_par)
    Nion_BO.initialize_GP(n_init)
    Nion_BO.run_optimization(n_optimize)
    Nion_BO.saveresults()
    end = time.time()
    print(f"Total {n_init} initialization measurements, {n_optimize} iterations, time elaspsed {float(end - start):.2f} sec.")

# option of two step improvements:
# step 1: fix up to C23 by minimizing emittance.
abr_activate = [True, True, True, True, True, True, True, False, False, False, False, False]
filename = 'C:/Users/ASUsers/Downloads/2ndorder_attempt00_45mrad_250ms_standardize_removebuffer_scale_stage1'
Nion_BO = BOinterface(abr_activate, option_standardize = option_standardize, aperture = aperture, CNNpath = CNNpath_1, 
filename = filename, exposure_t= exposure_t, remove_buffer= remove_buffer, apt_option= apt_option, acq_func_par = acq_func_par)
Nion_BO.initialize_GP(n_init)
Nion_BO.run_optimization(n_optimize)
Nion_BO.saveresults()

# step 2: refine first order aberrations by minimizing defocus.
abr_activate = [True, True, True, False, False, False, False, False, False, False, False, False]
filename = 'C:/Users/ASUsers/Downloads/2ndorder_attempt00_45mrad_250ms_standardize_removebuffer_scale_stage2'
Nion_BO = BOinterface(abr_activate, option_standardize = option_standardize, aperture = aperture, CNNpath = CNNpath_2, 
filename = filename, exposure_t= exposure_t, remove_buffer= remove_buffer, apt_option= apt_option, acq_func_par = acq_func_par)
Nion_BO.initialize_GP(n_init)
Nion_BO.run_optimization(n_optimize)
Nion_BO.saveresults()