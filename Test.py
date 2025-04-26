import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
# Path for PyNab package
sys.path.append("C:/Users/ricardo/Downloads/pyNab-master/pyNab-master/src")
# Path for deltarice (package created by David Matthew for Nab)
sys.path.append("C:/Users/ricardo/Downloads/deltarice-master/deltarice-master/build/lib.linux-x86_64-3.10")
import nabPy as Nab
import h5py

# Load in data
data = Nab.File("/mnt/c/Users/ricardo/Downloads/Run5320_0.h5")


# Extract coincidence waveforms
coinc = data.coincWaves()
pulsr = data.pulsrWaves()
noise = data.noiseWaves()
single = data.singleWaves()

'''params: list with varying elements depending on the method passed
        (optional parameters shown in parenthesis)
        'trap': [risetime, flat top length, decay rate, (threshold percent, mean, shift)]
        'cusp': [risetime, flat top length, decay rate, (threshold percement, mean, shift)]
        'doubletrap': [risetime, flat top length, decay rate, (threshold percent, mean, shift)]'''

filter_settings = [1250, 50, 1250]

# Coincidence waves energy timings
Stimings = single.determineEnergyTiming(method='trap', params=filter_settings)
print(Stimings)

# Plotting energies of coincidence waves
fig, ax = plt.subplots(figsize(10, 5))
fig.patch.set_facecolor('xkcd:white')
fig.text(0.5, -0.05, "Generated with: single.determineEnergyTiming(method='trap', params=filter_settings)", ha='center')
ax.set_xlabel('ADC Channel')
ax.set_ylabel('Counts')
ax.grid(True)
ax.set_title('Energy Histogram \n Singles Data')
Stimings.hist('energy', bins = Nab.np.arange(0,6000))
plt.xlim(0,6000)
plt.ylim(0,20)
Stimings.data().columns



# Print something to show that the code has resolved completely.
print("Test successful!")
