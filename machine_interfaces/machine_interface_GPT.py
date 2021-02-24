import numpy as np
import sys
sys.path.insert(1, '/home/chenyu/Desktop/GaussianProcess/GPTrelated/')
from uscope_calc import sim
import matplotlib.pyplot as plt
import os
import time
# CNN related libraries
from keras import applications, optimizers, callbacks
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

class machine_interface:
    def __init__(self, dev_ids, start_point = None):
        self.pvs = np.array(dev_ids)
        self.name = 'GPT' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        self.CNNmodel = self.loadCNN('CNNmodels/VGG16.h5'); # hard coded model path for now
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def loadCNN(self, path):
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

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=1)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

    def getState(self): 
        ASCIIFILE = '/home/chenyu/Desktop/GaussianProcess/outscope.txt'
        PNGFILE = '/home/chenyu/Desktop/GaussianProcess/ronchigram.npy'
        MConHBAR  =  2.59e12
        maxsig = 1  # determine how many standard deviations are we going to plot

        xlim, ylim, shadow = sim(
                H1    = self.x[0][0],
                H2    = self.x[0][0] + self.x[0][1],
                S1 = 2.5e5,
                S2 = 2.5e5,
                S3 = 119931.5,
                S4 = 648691.415,
                # S6 = 390000,
                # S7 = -654100.0,
                # S1 = self.x[0][2],  #2.5e5,
                # S2 = self.x[0][3],  #2.5e5,
                # S3 = self.x[0][4],  #119931.5,
                # S4 = self.x[0][5],  #648691.415,
                S6 = self.x[0][2],  #390000,
                S7 = self.x[0][3],  #-654100.0
                alpha = 1.0e-4*5,
                Obj = self.x[0][4],
                # Obj=-3.7505e6, # new objective lens setting with high conv angle
                # Obj=-9.39e5,
             )      # the parameters that are not given an value here would be set to the default values, which could be found in uscope.py
                    # the sim function would return the Ronchigram, and save the outscope.txt file to the path that was calling this function
                    # i.e. the path of the Jupyte Notebook


        # Get emittance from CNN model using the shadow returned by GPT
        x_list = []
        frame = self.scale_range(shadow, 0, 1)
        new_channel = np.zeros(frame.shape)
        img_stack = np.dstack((frame, new_channel, new_channel))
        x_list.append(img_stack)
        x_list = np.concatenate([arr[np.newaxis] for arr in x_list])
        prediction = self.CNNmodel.predict(x_list, batch_size = 1)
        print(prediction)
        objective_state = -prediction[0][0]
        del x_list, img_stack, frame, prediction

        # # Get emittance from electron profiles in the GPT output
        # # check whether outscope file is ready in the path defined above
        # if ~os.path.exists(ASCIIFILE):
        #     time.sleep(1)
        # # time.sleep(10)
        # # number of pixels that will be used to generate x-y and kx-ky grid
        # N = 40

        # # process the simulated results from outscope.txt, then remove the file
        # screen =  np.loadtxt(ASCIIFILE, skiprows=5)
        
        # x  = screen[:,0]
        # y  = screen[:,1]
        # x = x * 1e12
        # y = y * 1e12  # x and y in unit of pm

        # ax = np.divide(screen[:,4], screen[:,6])
        # ay = np.divide(screen[:,5], screen[:,6])
        # arx = np.sqrt(ax**2 + ay**2)
        # index = np.where(arx < 0.04)

        # x = x[index]
        # y = y[index]
        # ax = ax[index]
        # ay = ay[index]

        # # directly calculate emittance from defination for all the simulated electrons
        # emit_1 = np.average(x**2 + y**2)
        # emit_2 = np.average(ax**2 + ay**2)
        # emit_3 = np.average(x*ax + y*ay)
        # emit = np.sqrt(emit_1 * emit_2 - emit_3**2) # emittance in unit of [pm*rad]

        # # return objective state as the negative sum of emittance
        # # negative sum of emit is used as the BO will maximize the objective state, as a result of using the negative UCB as acquisition func
        # objective_state = -emit   
        # # print(objective_state)

        # The rest is the same for two different emittance calculation methods
        np.save(PNGFILE, shadow)
        # # save Ronchigram figure as a reference of tuning
        # # fig = plt.figure()
        # # plt.imshow(shadow)
        # # plt.savefig('ronchigram.png')
        # # os.remove(ASCIIFILE)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
