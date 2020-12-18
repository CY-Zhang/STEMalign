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
        self.x = np.array(x_new, ndmin=2)
        # add expressions to set machine ctrl pvs to the position called self.x -- Note: self.x is a 2-dimensional array of shape (1, ndim). To get the values as a 1d-array, use self.x[0]

    def getState(self): 
        ASCIIFILE = '/home/cz489/STEMalign_BO/outscope.txt'
        MConHBAR  =  2.59e12
        maxsig = 2  # determine how many standard deviations are we going to plot

        sim(
             H1    = self.x[0][0], #0.2, #-2.9,
             H2    = self.x[0][1], #0.2, #-2.9,
             S0    = 0.0,
             Obj   = 1.30306, #1.30305,
             alpha = 2.0e-5, #1.1e-5
             seed  = 0,
             erL   = 0.0,
             erTh  = 0.0
             )
        print('simulation finished')

        # check whether outscope file is ready
        if ~os.path.exists(ASCIIFILE):
            time.sleep(1)

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

        N = 24

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
        k_abs = np.power(np.power(kx_grid, 2) + np.power(ky_grid, 2), 0.5)

        # calculate emittance
        emit_1 = np.power(kx_grid, 2) + np.power(ky_grid, 2)  # first term, absolute value of gradient of aberration function
        emit_2 = np.power(x_grid, 2) + np.power(y_grid, 2)
        emit_3 = kx_grid * x_grid + ky_grid * y_grid  # third term, cross term between x/y and gradient along x/y
        emit = emit_1 * emit_2 - emit_3

        # return objective state as teh sum of emittance
        objective_state = np.log(np.sum(emit[~np.isnan(kx_grid)]))
        # print(objective_state)

        # save Ronchigram figure as a reference of tuning
        # fig = plt.figure()
        # plt.imshow(shadow)
        # plt.savefig('ronchigram.png')
        # os.remove(ASCIIFILE)

        return np.array(self.x, ndmin = 2), np.array([[objective_state]])
    
    
