import sys
sys.path.insert(1, '/home/chenyu/Desktop/git/STEMalign/')
from Bayesian_optimization import BOinterface

abr_activate = [True, False, True, False, False, False, False, False, False, False, False, False]   # activated aberration coefficients
CNNpath = '/home/chenyu/Desktop/git/STEMalign/TorchModels/Test18_Nion_2ndorder_45mradApt_50mradLimit_emit+defocus_Adam_attempt02.pt'
Nion_BO = BOinterface(abr_activate, option_standardize = False, aperture = 0, CNNpath = CNNpath)
Nion_BO.initialize_GP(10)
Nion_BO.run_optimization(10)