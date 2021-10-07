import os
import math
import pickle
from math import cos, sin
import numpy as np
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

### filenames, constants ###

PATH = "/home/chenyu/Desktop/git/STEMalign/GPT_misalignment/"
GDFFILE   = "{}temp.gdf" .format(PATH)
ASCIIFILE = "{}outscope.txt".format(PATH)
TRANSFILE = "{}trans.gdf".format(PATH)
TRANSASCII = "{}trans.txt".format(PATH)
ROOT      = "/home/chenyu/Software/gpt310x64_20210921/bin/"
EXE       = "{}gpt".format(ROOT)
EXETXT    = "{}gdf2a".format(ROOT)
EXETRANS  = "{}gdftrans".format(ROOT)
MConHBAR  =  2.59e12 #inverse meters
# sampleL = 5e-10
sampleL = 2.096e-9*4
sampleL = 9e-9
sampleL = 32 * 0.396e-9
sampleScale = 1
errorsigmaL = 0
errorsigmaTheta = 0.0
maxsig = 1.0
#H1 = 1228.5
#S6 = 390000
#S7 =-680186.0
#Obj=-12180368.5

params = {"sol1nI"  :   2.5e5,
          "sol1cH"  :   0.0,
          "sol1cV"  :   0.0,
          "sol2nI"  :   2.5e5,
          "sol2cH"  :   0.0,
          "sol2cV"  :   0.0,
          "hex1G"   :   899,
          "hex1cH"  :   0.0,
          "hex1cV"  :   0.0,
          "soltnI"  :   1.199315e5,
          "soltcH"  :   0.0,
          "soltcV"  :   0.0,
          "csol1nI" :   6.48691415e5,
          "csol1cH" :   0.0,
          "csol1cV" :   0.0,
          "csol2nI" :  -6.48691415e5,
          "csol2cH" :   0.0,
          "csol2cV" :   0.0,
          "hex2G"   :   899,
          "hex2cH"  :   0.0,
          "hex2cV"  :   0.0,
          "csol3nI" :   3.9e5,
          "csol3cH" :   0.0,
          "csol3cV" :   0.0,
          "csol4nI" :  -6.541e5,
          "csol4cH" :   0.0,
          "csol4cV" :   0.0,
          "sol3nI"  :  -9.39e5,
          "sol3cH"  :   0.0,
          "sol3cV"  :   0.0,
          "sol4nI"  :   0.0,
          "alpha"   :   1e-4,
          "theta"   :   0.0,
          "delta"   :   0.0}

eleprefix = ["sol1",  "sol2",  "solt","hex1",
             "csol1", "csol2", "hex2",
             "csol3", "csol4", "sol3"]

selected = [False, False, False, False, 
            False, False, False, 
            False, False, False]

elepostfix = ["ox", "oy", "oz",
              "xx", "xy", "xz",
              "yx", "yy", "yz"]

errornames = [[pre + post for post in elepostfix] for pre in eleprefix]

### wrapper for gpt ###

def sim(S1    = params["sol1nI"],
        S1CH  = params["sol1cH"], 
        S1CV  = params["sol1cV"], 
        S2    = params["sol2nI"],
        S2CH  = params["sol2cH"],
        S2CV  = params["sol2cV"], 
        S3    = params["soltnI"],
        S3CH  = params["soltcH"],
        S3CV  = params["soltcV"], 
        H1    = params["hex1G"],
        H1CH  = params["hex1cH"],
        H1CV  = params["hex1cV"],
        S4    = params["csol1nI"],
        S4CH  = params["csol1cH"],
        S4CV  = params["csol1cV"],
        S5    = params["csol2nI"],
        S5CH  = params["csol2cH"],
        S5CV  = params["csol2cV"],
        H2    = params["hex2G"],
        H2CH  = params["hex2cH"],
        H2CV  = params["hex2cV"],
        S6    = params["csol3nI"],
        S6CH  = params["csol3cH"],
        S6CV  = params["csol3cV"],
        S7    = params["csol4nI"],
        S7CH  = params["csol4cH"],
        S7CV  = params["csol4cV"],
        Obj   = params["sol3nI"],
        ObjCH = params["sol3cH"],
        ObjCV = params["sol3cV"],
        S9    = params["sol4nI"],
        alpha = params["alpha"],
        theta = params["theta"],
        delta = params["delta"],
        seed  = 0,
        erL   = errorsigmaL,
        erTh  = errorsigmaTheta):
    np.random.seed(seed = seed)
    rs     = [np.random.normal(size = 6) for dummy in range(0, len(eleprefix))]
    errors = []
    for i in range(len(rs)):
        r = rs[i]
        # errors.append([1e-5,1e-5,0,1,0,0,0,1,0]) if selected[i] else errors.append([0,0,0,1,0,0,0,1,0])
        errors.append([r[0]*erL, r[1]*erL, r[2]*erL,
            cos(r[3]*erTh)*cos(r[5]*erTh) - cos(r[4]*erTh)*sin(r[3]*erTh)*sin(r[5]*erTh),
            -cos(r[3]*erTh)*sin(r[5]*erTh) - cos(r[4]*erTh)*cos(r[5]*erTh)*sin(r[3]*erTh),
            sin(r[3]*erTh)*sin(r[4]*erTh),
            cos(r[5]*erTh)*sin(r[3]*erTh) + cos(r[3]*erTh)*cos(r[4]*erTh)*sin(r[5]*erTh),
            cos(r[3]*erTh)*cos(r[4]*erTh)*cos(r[5]*erTh) - sin(r[3]*erTh)*sin(r[5]*erTh),
            -cos(r[3]*erTh)*sin(r[4]*erTh)]) if selected[i] else errors.append([0,0,0,1,0,0,0,1,0])

    # errors = [[r[0]*erL, r[1]*erL, r[2]*erL,
    #            cos(r[3]*erTh)*cos(r[5]*erTh) - cos(r[4]*erTh)*sin(r[3]*erTh)*sin(r[5]*erTh),
    #           -cos(r[3]*erTh)*sin(r[5]*erTh) - cos(r[4]*erTh)*cos(r[5]*erTh)*sin(r[3]*erTh),
    #            sin(r[3]*erTh)*sin(r[4]*erTh),
    #            cos(r[5]*erTh)*sin(r[3]*erTh) + cos(r[3]*erTh)*cos(r[4]*erTh)*sin(r[5]*erTh),
    #            cos(r[3]*erTh)*cos(r[4]*erTh)*cos(r[5]*erTh) - sin(r[3]*erTh)*sin(r[5]*erTh),
    #           -cos(r[3]*erTh)*sin(r[4]*erTh)] for r in rs] 

    cmdA = "{} -o {} {}hexuscope.in {}{}".format(EXE, GDFFILE, PATH,
          "".join(["{}={} ".format(x,y) for x, y in zip(params.keys(), 
          [S1, S1CH, S1CV, S2, S2CH, S2CV, H1, H1CH, H1CV, S3, 
           S3CH, S3CV, S4, S4CH, S4CV, 
           S5, S5CH, S5CV, H2, H2CH, H2CV, S6, S6CH, S6CV, S7, S7CH, S7CV, Obj, ObjCH, ObjCV, S9, alpha, theta, delta])]), 
          "".join(["{}={} ".format(s, t) for x, y in zip(errornames, errors) for s, t in zip(x, y)]))
    cmdC = "{} -o {} {} time x y z G".format(EXETRANS, TRANSFILE, GDFFILE)

    cmdB = "{} -o {} {}".format(EXETXT, ASCIIFILE, GDFFILE)

    cmdD = "{} -o {} {}".format(EXETXT, TRANSASCII, TRANSFILE)
    
    # cmdA,C,D to track the particles, cmdA,B to run standard screen
    os.system(cmdA)
    os.system(cmdC)
    # os.system(cmdB)
    os.system(cmdD)
    screen =  np.loadtxt(ASCIIFILE, skiprows=5)
    
    x  = screen[:,0]
    y  = screen[:,1]
    kx = np.divide(screen[:,4], screen[:,6])
    ky = np.divide(screen[:,5], screen[:,6])

    meankx = np.mean(kx)
    sigkx  = np.std(kx)

    meanky = np.mean(ky)
    sigky  = np.std(ky)
    
    N = 40
    # set a fixed kx, ky limit if necessary
    sigkx = 0.040 / maxsig
    sigky = 0.040 / maxsig

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

    # Remove possible nan points that would make following interpolation step fail
    y_grid[np.isnan(y_grid)]=0
    x_grid[np.isnan(x_grid)]=0
    index = np.where(x_grid != 0)

    xfunc = interpolate.SmoothBivariateSpline(kx_grid[index].flatten(), ky_grid[index].flatten(), x_grid[index].flatten(), kx=5, ky=5)
    yfunc = interpolate.SmoothBivariateSpline(kx_grid[index].flatten(), ky_grid[index].flatten(), y_grid[index].flatten(), kx=5, ky=5)

    ky_fine = np.linspace(-sigkx*maxsig, sigkx*maxsig, 128)
    kx_fine = np.linspace(-sigkx*maxsig, sigkx*maxsig, 128)

    FILENAME = "/home/chenyu/Desktop/git/STEMalign/GPTrelated/trnsmssn_antialiasing.pickle"

    with open(FILENAME, "rb") as f:
        trnsmssn = pickle.load(f)

    shadow = np.array([[trnsmssn((xfunc(kx, ky)[0][0]/sampleScale + sampleL/2)%sampleL, 
                                 (yfunc(kx, ky)[0][0]/sampleScale + sampleL/2)%sampleL)[0] for kx in kx_fine] for ky in ky_fine])
    return maxsig*sigkx, maxsig*sigky, shadow

