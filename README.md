# STEMalign
Automatic STEM alignment with Bayesian optimization based on BoTorch, GP model from GPyTorch

Current libraries:

Python 3.7.3
CUDA 10.1
Tensorflow 2.2.0
Keras 2.3.1
botorch 0.4.0
torch 1.7.1 (CPU only)
gpytorch 1.4.2

09/21/21 Started new one based on boTorch, the old one using SLAC-GP can be found on https://github.com/CY-Zhang/Bayesian-optimization-using-Gaussian-Process.git
10/12/21 BO optimization on GPT simulation with misalignment, and BO optimization on Nion instruments are added. Nion part not tested on real instruments yet.