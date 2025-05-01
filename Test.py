import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import Paths


User = 'Skylar'


if User.lower() == 'skylar':
    paths = Paths.Skylar_Paths

elif User.lower() == 'arush':
    paths = Paths.Arush_Paths


# Path for PyNab package
sys.path.append(paths[0])
# Path for deltarice (package created by David Matthews for Nab)
sys.path.append(paths[1])
import nabPy as Nab
import h5py

'''
Currently just trying to replicate the Basic.ipynb file to make sure
everything works on this computer. Also to help learn the PyNab package,
the methods available in it, and how to use it overall.
'''


# This "try" statement is for users using WSL which has no graphical display
# On normal systems or with systems without tkinter, it will just pass instead.
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass # Fallback to system defaults


# Load in data
data = Nab.File(paths[2]+"Run5730_1.h5")



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



#-------------------------------------------------------------------------------

counts = {}
for i in range(10000):
    try:
        data = noise.wave(i)
        for reading in data:
            if reading in counts.keys():
                counts[reading] += 1
            else:
                counts[reading] = 1
    except:
        break

print(f'{i} waveforms recorded.')

x = list(counts.keys())
y = list(counts.values())

plt.plot(x,y, 'o')
plt.show()
sys.exit()



filter_settings = [1250, 50, 1250]

# Coincidence waves energy timings
Ctimings = coinc.determineEnergyTiming(method='trap', params=filter_settings)
print(Ctimings)

# Plotting energies of coincidence waves
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('xkcd:white')
fig.text(0.5, -0.05, "Generated with: coinc.determineEnergyTiming(method='trap', params=filter_settings)", ha='center')
ax.set_xlabel('ADC Channel')
ax.set_ylabel('Counts')
ax.grid(True)
ax.set_title('Energy Histogram \n Singles Data')
Ctimings.hist('energy', bins = Nab.np.arange(0,6000))
plt.xlim(0,6000)
plt.ylim(0,20)
Ctimings.data().columns
plt.show()



# Print something to show that the code has resolved completely.
print("Test successful!")
