import numpy as np
import sys
sys.path.insert(1, '/home/chenyu/Desktop/GaussianProcess/GPTrelated/')
from uscope_calc import sim
import matplotlib.pyplot as plt
import os
import time
# CNN related libraries
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import tensorflow as tf
from nion.utils import Registry

class machine_interface:

    def __init__(self, dev_ids, start_point = None, CNNoption = 0, CNNpath = ''):
        # initialize aberration list, this has to come before setting aberrations
        self.abr_list = ["C10", "C12.x", "C12.y", "C21.x", "C21.y", "C23.x", "C23.y", "C30", 
        "C32.x", "C32.y", "C34.x", "C34.y"]
        # Initialize stem controller
        self.stem_controller = Registry.get_component("stem_controller")
        for abr_coeff in self.abr_list:
            success, value = self.stem_controller.TryGetVal(abr_coeff)
            print(abr_coeff + ' successfully loaded.')
        
        # Connect to ronchigram camera and setup camera parameters
        self.ronchigram = self.stem_controller.ronchigram_camera
        frame_parameters = self.ronchigram.get_current_frame_parameters()
        frame_parameters["binning"] = 16
        frame_parameters["exposure_ms"] = 10
        self.ronchigram.start_playing(frame_parameters)
        self.ronchigram.stop_playing()

        os.environ["CUDA_VISIBLE_DEVICES"]="0" # specify which GPU to use
        self.pvs = np.array(dev_ids)
        self.name = 'Nion' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        self.CNNoption = CNNoption

        if CNNoption == 1:
            self.CNNmodel = self.loadCNN(CNNpath); # hard coded model path for now
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
              try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
              except RuntimeError as e:
                print(e)

        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def loadCNN(self, path):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        model = applications.VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
        print('Model loaded')
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.0))
        top_model.add(Dense(1,activation=None))
        new_model = Sequential()

        for l in model.layers:
            new_model.add(l)

        new_model.add(top_model)
        new_model.load_weights(path)
        return new_model

    def scale_range(self, input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    def aperture_generator(self, px_size, simdim, ap_size):
        x = np.linspace(-simdim, simdim, px_size)
        y = np.linspace(-simdim, simdim, px_size)
        xv, yv = np.meshgrid(x, y)
        apt_mask = mask = np.sqrt(xv*xv + yv*yv) < ap_size # aperture mask
        return apt_mask

    def setX(self, x_new):
        self.x = x_new
        idx = 0
        # print(x_new)
        # print(self.abr_list)
        # for nionswift-usim, change the aberration corrector status when calling setX
        for abr_coeff in self.abr_list:
            self.stem_controller.SetVal(abr_coeff, x_new[0][idx])
            idx += 1
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]
        return

    def getState(self): 
        # os.environ["CUDA_VISIBLE_DEVICES"]="0"
        self.ronchigram.start_playing()
        print("before grabbing frame.")
        frame = np.asarray(self.ronchigram.grab_next_to_start()[0])
        self.ronchigram.stop_playing()

        # Get emittance from CNN model using the shadow returned by GPT
        if self.CNNoption == 1:
            x_list = []
            # normalize then divided by 2 to match the contrast of Matlab simulated Ronchigrams
            # frame = self.scale_range(shadow, 0, 1) / 2 * self.aperture_generator(128, 40, 40)
            frame = self.scale_range(frame, 0, 1)
            new_channel = np.zeros(frame.shape)
            img_stack = np.dstack((frame, new_channel, new_channel))
            x_list.append(img_stack)
            x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
            prediction = self.CNNmodel.predict(x_list, batch_size = 1)
            # print(prediction)
            objective_state = 1 - prediction[0][0]
            print('Using CNN prediction.')
            del x_list, img_stack, frame, prediction

        # # Just for debug purpose, cannot run without CNN in this case.
        if self.CNNoption == 0:
            print('Running without CNN.')
            objective_state = 1
        # # print(objective_state)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])