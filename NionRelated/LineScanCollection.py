'''
06-17-21
Script to collect image stack with varying aberration coefficient.
Currently tested on simulator nionswift-sim.
'''

from nion.utils import Registry
import numpy as np

# define the path to save the file
path = 'C:/Users/Chenyu/Documents/GitHub/STEMalign/NionRelated/'
# number of steps to change the aberration coefficients.
nsteps = 20
# name of aberration coefficient to vary
abr_coeff = "C30"
# total range of aberration in m
abr_range = 5e-5
# camera parameters
exposure_ms = 10
binning = 8     # full frame has 2048 px.
# initialize list for aberration and image.
value_list = [(i - nsteps//2) * abr_range / nsteps for i in range(nsteps)]
image_stack = []
# Connect to stem controller to setup aberration
stem_controller = Registry.get_component("stem_controller")
success, value = stem_controller.TryGetVal(abr_coeff)
print(success)
# Connect to ronchigram camera and setup camera parameters
ronchigram = stem_controller.ronchigram_camera
frame_parameters = ronchigram.get_current_frame_parameters()
frame_parameters["binning"] = binning
frame_parameters["exposure_ms"] = exposure_ms
ronchigram.start_playing(frame_parameters)

# start acquisition for each aberration value in the list
for i in value_list:
    if stem_controller.SetVal(abr_coeff, i):
        image_stack.append(np.asarray(ronchigram.grab_next_to_start()[0]))
        print(abr_coeff + ' ' + str(i))
ronchigram.stop_playing()
stem_controller.SetVal(abr_coeff, 0)

# save the acquired image stack.
image_stack = np.asarray(image_stack)
filename = abr_coeff + '_' + str(abr_range) + 'm_' + str(nsteps) + 'steps_' + str(exposure_ms) + 'ms_bin' + str(binning) + '.npy'
np.save(path + filename, image_stack)