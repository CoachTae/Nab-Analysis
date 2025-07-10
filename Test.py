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



# This "try" statement is for users using WSL which has no graphical display
# On normal systems or with systems without tkinter, it will just pass instead.
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass # Fallback to system defaults



run = Nab.DataRun(paths[2], 8118)
parameters = run.parameterFile()


#noise = run.noiseWaves()
#regcoinc = run.coincWaves()
#coinc = run.coincWaves().headers()
singles = run.singleWaves()
singles_headers = run.singleWaves().headers()

# Filter for populated pixel (1061) indices
singles_indices = singles_headers[singles_headers['pixel'] == 1061].index.tolist()

# Cut for only waveforms from most populated pixel
singles.defineCut('custom', singles_indices)

# Initialize filter
KFilter = KF()

for index in singles_indices:
    coinc._waveformFile__waves[index] = Kfilter.smooth(coinc._waveformFile__waves[index])

# Apply trap filter and determine energies
filter_settings = [1250, 50, 1250]
singles_energies = singles.determineEnergyTiming('trap', params=filter_settings)

# Get headers object w/ energy column
energies_headers = singles_energies.data()

# Pull just the energies
energies = energies_headers['energy']*0.3

# Histogram
bin_size = 0.2
min_energy = energies.min()
max_energy = energies.max()
bins = np.arange(min_energy, max_energy + bin_size, bin_size)

plt.hist(energies, bins=bins)
plt.xlabel('Energy')
plt.ylabel('Count')
plt.show()


print("TEST COMPLETED!")
