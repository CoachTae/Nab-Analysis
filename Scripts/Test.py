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


# 7637 = background run
# 7900 = beam data
# 7932 = background data (but shows 2x coincidences compared to noise?)
# 7933 = beam data
# 8118 = source data


run = Nab.DataRun(paths[2], 7637)
#parameters = run.parameterFile()

singles = run.singleWaves()
coinc = run.coincWaves()
#coinc_headers = run.coincWaves().headers()
noise = run.noiseWaves()
#noise_headers = run.noiseWaves().headers()

#coinc.defineCut('hit type', '=', 2)
#noise.defineCut('pixel', '=', 1097)
#coinc.defineCut('pixel', '=', 1097)

print(len(singles.waves()))
print(len(coinc.waves()))
print(len(noise.waves()))
sys.exit()
for wave in coinc.waves():
    Plotting.plot_wf(wave)
