import os
import sys
import copy
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Paths
import Plotting
import json
from Classes.KalmanFilterClass import KF


start_time = time.time()


run_num = 7900


User = 'Skylar2'


if User.lower() == 'skylar':
    paths = Paths.Skylar_Paths

elif User.lower() == "skylar2":
    paths = Paths.Skylar_Home

elif User.lower() == 'arush':
    paths = Paths.Arush_Paths


# Path for PyNab package
sys.path.append(paths[0])
# Path for deltarice (package created by David Matthews for Nab)
sys.path.append(paths[1])
import nabPy as Nab
import basicFunctions as bf
import h5py


# This "try" statement is for users using WSL which has no graphical display
# On normal systems or with systems without tkinter, it will just pass instead.
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass # Fallback to system defaults



run = Nab.DataRun(paths[2], run_num)
parameters = run.parameterFile()


#noise = run.noiseWaves()
coinc = run.coincWaves()
coinc_headers = run.coincWaves().headers()
#singles = run.singleWaves()
#singles_headers = run.singleWaves().headers()

random_e_indices = coinc_headers[coinc_headers['hit type'] == 0].sample(n=10).index.tolist()

filter_settings = [1250, 50, 1250]
KFilter = KF()

reg_waves = []
filt_waves = []
for idx in random_e_indices:
    reg_waves.append(coinc.wave(idx))
    filt_waves.append(KFilter.smooth(coinc.wave(idx)))

reg_waves = np.array(reg_waves)
filt_waves = np.array(filt_waves)


trapped_waves = bf.applyTrapFilter(reg_waves, *filter_settings, returnfilter=True)
trapped_filt_waves = bf.applyTrapFilter(filt_waves, *filter_settings, returnfilter=True)

for i in range(len(trapped_waves[:,0])):
    reg_wave = trapped_waves[i,:]
    filt_wave = trapped_filt_waves[i,:]

    Plotting.plot_stacked_waveforms(reg_wave, filt_wave, label1="Raw Waveform", label2="Filtered Waveform")
