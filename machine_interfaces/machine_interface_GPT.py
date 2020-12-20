import numpy as np
import sys
sys.path.insert(1, '/home/cz489/STEMalign_BO/GPTrelated')
from uscope_calc import sim
import matplotlib.pyplot as plt
import os
import time

class machine_interface:
    def __init__(self, dev_ids, start_point = None):
        self.pvs = np.array(dev_ids)
        self.name = 'GPT' #name your machine interface. doesn't matter what you call it as long as it isn't 'MultinormalInterface'.
        if type(start_point) == type(None):
            current_x = np.zeros(len(self.pvs)) #replace with expression that reads current ctrl pv values (x) from machine
            self.setX(current_x)
        else: 
            self.setX(start_point)

    def setX(self, x_new):
        self.x = np.array(x_new, ndmin=1)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

    def getState(self): 
        ASCIIFILE = '/home/cz489/STEMalign_BO/outscope.txt'
        PNGFILE = '/home/cz489/STEMalign_BO/ronchigram.npy'
        MConHBAR  =  2.59e12
        maxsig = 1  # determine how many standard deviations are we going to plot
                    # use +/- 1 standard deviation for now, maxsig = 2 works well when the probe is bad, but could generate emply plot 
                    # when the probe is well focused (??)

        shadow = sim(
                H1    = self.x[0][0],
                H2    = 0.0,
                S0    = 0.0,
                S3    = 330668.75, # 3.3066875e5
                S4    = 330598.75, # 3.3066875e5
                S5    = -330668.75,
                S6    = -330668.75,
                Obj   = 1.30305, #1.30305,
                alpha = 2.0e-5, #1.1e-5
                seed  = 0,
                erL   = 0.0,
                erTh  = 0.0
             )      # the parameters that are not given an value here would be set to the default values, which could be found in uscope.py
                    # the sim function would return the Ronchigram, and save the outscope.txt file to the path that was calling this function
                    # i.e. the path of the Jupyte Notebook
                # H1    = self.x[0][0], #0.2, #-2.9,
                # H2    = self.x[0][1], #0.2, #-2.9,

        # check whether outscope file is ready in the path defined above
        if ~os.path.exists(ASCIIFILE):
            time.sleep(1)

        # number of pixels that will be used to generate x-y and kx-ky grid
        N = 24

        # build a circular mask
        x_grid, y_grid = np.meshgrid(np.linspace(-N/2, N/2, N),
                             np.linspace(-N/2, N/2, N))
        temp = x_grid * x_grid + y_grid * y_grid
        mask =temp < N*N/4

        # process the simulated results from outscope.txt, then remove the file
        screen =  np.loadtxt(ASCIIFILE, skiprows=5)
        
        x  = screen[:,0]
        y  = screen[:,1]
        kx = MConHBAR*screen[:,4]*screen[:,7]
        ky = MConHBAR*screen[:,5]*screen[:,7]

        meanx = np.mean(x)
        sigx  = np.std(x)

        meany = np.mean(y)
        sigy  = np.std(y)

        kx_bins = [[[] for n in range(0,N)] for m in range(0,N)]
        ky_bins = [[[] for n in range(0,N)] for m in range(0,N)]

        kx_grid = np.zeros([N, N])
        ky_grid = np.zeros([N, N])

        x_grid, y_grid = np.meshgrid(sigx*np.linspace(-maxsig, maxsig, N),
                                     sigy*np.linspace(-maxsig, maxsig, N))

        for xi, yi, kxi, kyi in zip(x, y, kx, ky):
            i = int(0.5*N*((yi-meany)/(maxsig*sigy)) + 0.5*N)
            j = int(0.5*N*((xi-meanx)/(maxsig*sigx)) + 0.5*N)
            if i < 0 or i > N-1 or j < 0 or j > N-1:
                continue
            kx_bins[i][j].append(kxi)
            ky_bins[i][j].append(kyi)

        for i in range(0, N):
            for j in range(0, N):
                kx_grid[i,j] = np.mean(kx_bins[i][j])
                ky_grid[i,j] = np.mean(ky_bins[i][j])
        # remove the points in the mesh grid that did not collect any electron, then perform linear fit between kx and 
        # this should guarantee a non-nan fitting results of a and b, but the fitting result might be inaccurate.
        idx = np.where(np.isnan(kx_grid[12,:])==False)
        a, b = np.polyfit(x_grid[12,idx][0], kx_grid[12,idx][0], 1)

        if np.isnan(a) or np.isnan(b):  # if the fitting fail
            return np.array(self.x, ndmin = 2), np.array([[np.inf]])


        kx_grid = kx_grid - a * x_grid
        ky_grid = ky_grid - a * y_grid
        k_abs = np.power(np.power(kx_grid, 2) + np.power(ky_grid, 2), 0.5)

        # calculate emittance
        emit_1 = np.power(kx_grid, 2) + np.power(ky_grid, 2)  # first term, absolute value of gradient of aberration function
        emit_2 = np.power(x_grid, 2) + np.power(y_grid, 2)
        emit_3 = kx_grid * x_grid + ky_grid * y_grid  # third term, cross term between x/y and gradient along x/y
        emit = emit_1 * emit_2 - emit_3

        # return objective state as teh sum of emittance
        emit[np.isnan(emit)] = 0
        objective_state = -np.sum(emit*mask)
        # print(objective_state)
        np.save(PNGFILE, shadow)
        # save Ronchigram figure as a reference of tuning
        # fig = plt.figure()
        # plt.imshow(shadow)
        # plt.savefig('ronchigram.png')
        os.remove(ASCIIFILE)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
