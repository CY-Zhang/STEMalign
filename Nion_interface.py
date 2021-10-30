# Nion instrument related libraries
from nion.utils import Registry
import numpy as np
import threading

class Nion_interface():
    '''
    # TODO: set the aberration limits to changeable values.
    Input:
    act_list: a boolean list that determines which aberration coefficent is variable. The length should be the same as self.abr_lsit.
    readDefault: boolean that determines whether read the initial aberration values. Not in use.
    detectorCenter: boolean that determines whether to automatically detect the center of the Rochigram. Would use the center of the whole frame if set to False.
    exposure_t: acquisition exposure time in ms.
    remove_buffer: boolean that determines whether to acquire two frames and discard the first one.
    '''
    def __init__(self, act_list = [], readDefault = False, detectCenter = False, exposure_t = 100, remove_buffer = True):

        # initialize aberration list, this has to come before setting aberrations, hard coded aberration limit for now.
        self.abr_list = ["C10", "C12.x", "C12.y", "C21.x", "C21.y", "C23.x", "C23.y", "C30","C32.x", "C32.y", "C34.x", "C34.y"]
        self.default = [2e-9, 2e-9, 2e-9, 20e-9, 20e-9, 20e-9, 20e-9, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6, 0.5e-6]
        self.abr_lim = [2e-6, 1.5e-7, 1.5e-7, 3e-6, 3e-6, 1e-5, 1e-5, 3e-4, 2e-4, 2e-4, 1.5e-4, 1.5e-4]
        self.activate = act_list

        # option to read existing default value, can be used when running experiment
        self.readDefault = readDefault
        self.aperture = 0
        self.remove_buffer = remove_buffer

        # Initialize stem controller
        self.stem_controller = Registry.get_component("stem_controller")
        for i in range(len(self.abr_list)):
            abr_coeff = self.abr_list[i]
            _, val= self.stem_controller.TryGetVal(abr_coeff)
            if self.readDefault:
                self.default[i] = val
            print(abr_coeff + ' successfully loaded.')

        # Connect to ronchigram camera and setup camera parameters
        self.ronchigram = self.stem_controller.ronchigram_camera
        frame_parameters = self.ronchigram.get_current_frame_parameters()
        frame_parameters["binning"] = 1
        frame_parameters["exposure_ms"] = exposure_t
        self.ronchigram.start_playing(frame_parameters)

        # Acquire a test frame to set the crop region based on center detected using COM.
        # TODO: besides the center position, also detect the side length to use.
        temp = np.asarray(self.ronchigram.grab_next_to_start()[0])
        if detectCenter:
            x = np.linspace(0, temp.shape[1], num = temp.shape[1])
            y = np.linspace(0, temp.shape[0], num = temp.shape[0])
            xv, yv = np.meshgrid(x, y)
            self.center_x = int(np.average(xv, weights = temp))
            self.center_y = int(np.average(yv, weights = temp))
        else:
            self.center_x = temp.shape[0] / 2
            self.center_y = temp.shape[1] / 2

        # Allocate empty array to save the frame acquired from camera
        self.size = 128
        self.frame = np.zeros([self.size, self.size])

    '''
    Method to scale a single frame to the range of [min, max].
    Input:
    img: a 2D numpy array saving the image to normalize.
    min: target min value after normalization
    max: target max value after normalization
    Output:
    img: normalized 2D numpy array.

    Option1: scale_range: directly normalize the image based on the max and min value on the image. This could lead to reduced
    contrast as the area outside aperture is much darker than inside the aperture.
    Option2: scale_range_aperture: this option normalize the image bsed on the max and min value within certain annular range.
    '''
    def scale_range(self, img, min, max):
        img += -(np.min(img))
        img /= np.max(img) / (max - min)
        img += min
        return img

    def scale_range_aperture(self, img, min, max):
        mask = self.aperture_generator(128, 50, 40)
        array = np.ndarray.flatten(img[np.where(mask==1)])
        img = img - np.amin(array)
        img = img / (np.amax(array) - np.amin(array))
        img = img * (max - min)
        img += min
        img[np.where(img < min)] = min
        return img

    def aperture_generator(self, px_size, simdim, ap_size):
        x = np.linspace(-simdim, simdim, px_size)
        y = np.linspace(-simdim, simdim, px_size)
        xv, yv = np.meshgrid(x, y)
        apt_mask = mask = np.sqrt(xv*xv + yv*yv) < ap_size # aperture mask
        return apt_mask

    '''
    function that set the values of activated aberration coefficients.
    Input:
    x_new: the list with new aberration values to set.
    '''
    def setX(self, x_new):
        self.x = x_new
        idx = 0
        idx_activate = 0
        # set activated aberration coeff to desired value, and default values for the rest
        for abr_coeff in self.abr_list:
            if self.activate[idx]:
                val = x_new[idx_activate] * self.abr_lim[idx] - self.abr_lim[idx] / 2    
                idx_activate += 1
            else:
                val = self.default[idx]
            # print(abr_coeff, val)
            self.stem_controller.SetVal(abr_coeff, val)
            idx += 1
        return

    '''
    function to acquire a single frame by calling grab_next_to_start.
    '''

    def acquire_frame(self):
        self.frame = np.zeros([self.size, self.size])
        # self.ronchigram.start_playing()
        # print('Acquiring frame')

        # if remove_buffer option is on, grab a frame without saving it.
        if self.remove_buffer:
            print("remove buffer")
            self.ronchigram.grab_next_to_start()
        temp = np.asarray(self.ronchigram.grab_next_to_start()[0])
        temp = temp[self.center_y - 640 : self.center_y + 640, self.center_x - 640: self.center_x + 640]
        new_shape = [self.size, self.size]
        shape = (new_shape[0], temp.shape[0] // new_shape[0],new_shape[1], temp.shape[1] // new_shape[1])
        temp = temp.reshape(shape).mean(-1).mean(1)
        self.frame = temp
        return

    '''
    function to stop acquisition on the Ronchigram camera.
    '''
    def stopAcquisition(self):
        if self.ronchigram:
            self.ronchigram.stop_playing()
        return