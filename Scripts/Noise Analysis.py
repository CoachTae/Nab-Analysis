import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/mnt/s/Python/Projects/Nab-Analysis")
import Paths
import Plotting
import json
import simulate_waveforms as SimWF

start_time = time.time()
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
import h5py
import basicFunctions as bf



# This "try" statement is for users using WSL which has no graphical display
# On normal systems or with systems without tkinter, it will just pass instead.
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass # Fallback to system defaults



#run = Nab.DataRun(paths[2], 7900)
#parameters = run.parameterFile()

#coinc = run.coincWaves()
#coinc_headers = run.coincWaves().headers()
#noise = run.noiseWaves()
#noise_headers = run.noiseWaves().headers()


wf = SimWF.sim_wf(3500, 3500, 12, 1250)[1]
wf = np.array([wf])
energies = []
for rise_time in range(50, 1550, 50):
    for top in range(10, 110, 10):
        energy, _ = bf.applyDoubleTrapFilter(wf, rise_time, top, 1250)
        #if energy < 75 or energy > 85:
            #print(f'rise_time: {rise_time}')
            #print(f'top: {top}')
        energies.append(float(energy))
print(f'Minimum: {np.min(energies)}')
print(f'Maximum: {np.max(energies)}')
print(f'Mean: {np.mean(energies)}')
print(f'SD: {np.std(energies)}')
