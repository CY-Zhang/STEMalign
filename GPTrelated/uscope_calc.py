import os
import math
import pickle
from math import cos, sin
import numpy as np
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

### filenames, constants ###

GDFFILE   = "/home/cz489/STEMalign_BO/temp.gdf" 
ASCIIFILE = "/home/cz489/STEMalign_BO/outscope.txt"
EXE       = "/nfs/acc/temp/cjd257/gpt310x64/bin/gpt"
EXETXT    = "/nfs/acc/temp/cjd257/gpt310x64/bin/gdf2a"
MConHBAR  =  2.59e12 #inverse meters
sampleL = 5e-10;
errorsigmaL = 1e-5
errorsigmaTheta = 1e-5
maxsig = 1

params = {"sol0nI"  :  -1.3e6,
          "sol1nI"  :   1.0e6,
          "sol1cH"  :   0.0,
          "sol1cV"  :   0.0,
          "sol2nI"  :  -2.0e6,
          "sol2cH"  :   0.0,
          "sol2cV"  :   0.0,
          "hex1G"   :  25.0,
          "csol1nI" :   3.3066875e5,
          "csol1cH" :   0.0,
          "csol1cV" :   0.0,
          "csol2nI" :   3.3066875e5,
          "csol2cH" :   0.0,
          "csol2cV" :   0.0,
          "hex2G"   :  -25.0,
          "csol3nI" :  -3.3066875e5,
          "csol3cH" :   0.0,
          "csol3cV" :   0.0,
          "csol4nI" :  -3.3066875e5,
          "csol4cH" :   0.0,
          "csol4cV" :   0.0,
          "sol3nI"  :   1.3031,
          "sol3cH"  :   0.0,
          "sol3cV"  :   0.0,
          "alpha"   :   1e-6}

eleprefix = ["sol1",  "sol2",  "hex1",
             "csol1", "csol2", "hex2",
             "csol3", "csol4", "sol3"]

elepostfix = ["ox", "oy", "oz",
              "xx", "xy", "xz",
              "yx", "yy", "yz"]

errornames = [[pre + post for post in elepostfix] for pre in eleprefix]

### wrapper for gpt ###

def sim(S0    = params["sol0nI"],
        S1    = params["sol1nI"],
        S1CH  = params["sol1cH"], 
        S1CV  = params["sol1cV"], 
        S2    = params["sol2nI"],
        S2CH  = params["sol2cH"],
        S2CV  = params["sol2cV"], 
        H1    = params["hex1G"],
        S3    = params["csol1nI"],
        S3CH  = params["csol1cH"],
        S3CV  = params["csol1cV"],
        S4    = params["csol2nI"],
        S4CH  = params["csol2cH"],
        S4CV  = params["csol2cV"],
        H2    = params["hex2G"],
        S5    = params["csol3nI"],
        S5CH  = params["csol3cH"],
        S5CV  = params["csol3cV"],
        S6    = params["csol4nI"],
        S6CH  = params["csol4cH"],
        S6CV  = params["csol4cV"],
        Obj   = params["sol3nI"],
        ObjCH = params["sol3cH"],
        ObjCV = params["sol3cV"],
        alpha = params["alpha"],
        seed  = 0,
        erL   = errorsigmaL,
        erTh  = errorsigmaTheta):
    np.random.seed(seed = seed)
    rs     = [np.random.normal(size = 6) for dummy in range(0, len(eleprefix))]
    errors = [[r[0]*erL, r[1]*erL, r[2]*erL,
               cos(r[3]*erTh)*cos(r[5]*erTh) - cos(r[4]*erTh)*sin(r[3]*erTh)*sin(r[5]*erTh),
              -cos(r[3]*erTh)*sin(r[5]*erTh) - cos(r[4]*erTh)*cos(r[5]*erTh)*sin(r[3]*erTh),
               sin(r[3]*erTh)*sin(r[4]*erTh),
               cos(r[5]*erTh)*sin(r[3]*erTh) + cos(r[3]*erTh)*cos(r[4]*erTh)*sin(r[5]*erTh),
               cos(r[3]*erTh)*cos(r[4]*erTh)*cos(r[5]*erTh) - sin(r[3]*erTh)*sin(r[5]*erTh),
              -cos(r[3]*erTh)*sin(r[4]*erTh)] for r in rs] 
    cmdA = "{} -o {} /home/cz489/STEMalign_BO/GPTrelated/hexuscope.in {}{}".format(EXE, GDFFILE, 
          "".join(["{}={} ".format(x,y) for x, y in zip(params.keys(), 
          [S0, S1, S1CH, S2CV, S2, S2CH, S2CV, H1, S3, S3CH, S3CV, S4, S4CH, S4CV, H2, S5, S5CH, S5CV, S6, S6CH, S6CV, Obj, ObjCH, ObjCV, alpha])]), 
          "".join(["{}={} ".format(s, t) for x, y in zip(errornames, errors) for s, t in zip(x, y)]))
    cmdB = "{} -o {} {}".format(EXETXT, ASCIIFILE, GDFFILE)
    if os.path.exists(ASCIIFILE):
      os.remove(ASCIIFILE)
    os.system(cmdA)
    os.system(cmdB)
    # print(ASCIIFILE)
    screen =  np.loadtxt(ASCIIFILE, skiprows=5)
    
    x  = screen[:,0]
    y  = screen[:,1]
    kx = MConHBAR*screen[:,4]*screen[:,7]
    ky = MConHBAR*screen[:,5]*screen[:,7]

    meankx = np.mean(kx)
    sigkx  = np.std(kx)

    meanky = np.mean(ky)
    sigky  = np.std(ky)

    N = 24

    x_bins = [[[] for n in range(0,N)] for m in range(0,N)]
    y_bins = [[[] for n in range(0,N)] for m in range(0,N)]

    x_grid = np.zeros([N, N])
    y_grid = np.zeros([N, N])

    kx_grid, ky_grid = np.meshgrid(sigkx*np.linspace(-maxsig, maxsig, N),
                                 sigky*np.linspace(-maxsig, maxsig, N))

    for xi, yi, kxi, kyi in zip(x, y, kx, ky):
        i = int(0.5*N*((kyi-meanky)/(maxsig*sigky)) + 0.5*N)
        j = int(0.5*N*((kxi-meankx)/(maxsig*sigkx)) + 0.5*N)
        if i < 0 or i > N-1 or j < 0 or j > N-1:
            continue
        x_bins[i][j].append(xi)
        y_bins[i][j].append(yi)

    for i in range(0, N):
        for j in range(0, N):
            x_grid[i,j] = np.mean(x_bins[i][j])
            y_grid[i,j] = np.mean(y_bins[i][j])

    xfunc = interpolate.SmoothBivariateSpline(kx_grid.flatten(), ky_grid.flatten(), x_grid.flatten())
    yfunc = interpolate.SmoothBivariateSpline(kx_grid.flatten(), ky_grid.flatten(), y_grid.flatten())

    ky_fine = np.linspace(-sigkx*maxsig, sigkx*maxsig, 201)
    kx_fine = np.linspace(-sigkx*maxsig, sigkx*maxsig, 201)

    FILENAME = "/home/cz489/STEMalign_BO/GPTrelated/trnsmssn.pickle"

    with open(FILENAME, "rb") as f:
        trnsmssn = pickle.load(f)

    shadow = np.array([[trnsmssn(xfunc(kx, ky)[0][0]%sampleL, yfunc(kx, ky)[0][0]%sampleL)[0] for kx in kx_fine] for ky in ky_fine])
    return shadow

