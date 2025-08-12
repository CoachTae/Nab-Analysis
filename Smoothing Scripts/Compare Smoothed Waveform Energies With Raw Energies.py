import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Paths
import json

# Get the parent directory so we can import KalmanFilterClass
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from Classes.KalmanFilterClass import KF


num_particles = 10
particle_type = 0 # 0 for protons, 2 for electrons


start_time = time.time()
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

# Load Data
run = Nab.DataRun(paths[2], 7616)
print(f'File loaded: {time.time() - start_time} seconds')

# Load coincidence waveforms
coinc = run.coincWaves()
coinc_headers = coinc.headers()
print(f'Waveforms pulled: {time.time() - start_time} seconds')


# Filter by particle type
    # coinc_headers['hit type'] == particle_type creates a list of booleans where, if the header matches particle_type, we get a True value, if not, False
    # coinc_headers[(boolean list)] only pulls the rows whose indices match with a "True" index in the given list
        # For example, [True, False, True] will only return rows 1 and 3 from coinc_headers
        # This is like our own personal way of doing .defineCut.
    # .iloc[] allows you to index panda arrays similar to lists, by picking which indexes you want to look at
    # We choose the indices of the first (num_particles) from coinc_headers[(boolean list)] 
electron_hits = coinc_headers[coinc_headers['hit type'] == particle_type].iloc[:num_particles]

# Select the timestamp of the last particle to survive both our hit type criteria and our first (num_particles) criteria
cutoff_timestamp = electron_hits['timestamp'].iloc[-1]

# This creates a list of indices for each particle that survives our criteria
    # Again, coinc_headers['hit type'] == particle_type creates a boolean list of where our particles are in the table
    # We select only the ones that pass that criteria (returned True from our list)
    # From the survivors, we select the first (num_particles) for efficiency
    # .index[] returns the indices of those first (num_particles) survivors
    # .tolist() turns that index return into a list
electron_hits = coinc_headers[coinc_headers['hit type'] == particle_type].index[:num_particles].tolist()


# Cut out non-(particle_type) particles and also cut out any particles that came after our selected one
    # To reiterate, we cut out more just for the sake of speed, and we choose timestamp because it's a convenient choice
coinc.defineCut('hit type', '=', particle_type)
coinc.defineCut('timestamp', '<=', cutoff_timestamp)

# Determine raw energies based on our cuts
coinc_eners = coinc.determineEnergyTiming(method='trap', params=[1250, 50, 1250], useGPU=True, batchsize=100)
print(f'Raw energies found: {time.time() - start_time} seconds')

# Create the filter
KFilter = KF()
# This is an arbitrary number modified to try to find best results
KFilter.set_transition_covariance(0.001)


# Reach into the private "__waves" attribute and only apply the filter to the ones that survived our criteria from earlier
print("Starting filter processing...")
for i in range(num_particles):
    print(f'Smoothing waveform {i+1} / {num_particles}: {time.time() - start_time} seconds.')
    coinc._waveformFile__waves[electron_hits[i]] = KFilter.smooth(coinc._waveformFile__waves[electron_hits[i]])

# Calculate new energies (notice it's the same coinc object, but with the specified waveforms in __waves having been run through the filter
smoothed_eners = coinc.determineEnergyTiming(method='trap', params=[1250, 50, 1250], useGPU=True, batchsize=100)
print(f'Smoothed energies found: {time.time() - start_time} seconds')


print("Plotting...")
fig, ax = plt.subplots()

# Determine the relative residuals
    # Positive number means the filter read out a higher energy than originally
    # Negative number means the filter read out a lower energy than originally
    # Multiply relative residual by 100 to get the percentage difference between the two.
print((smoothed_eners.data()['energy'] - coinc_eners.data()['energy'])/coinc_eners.data()['energy'])

# Plot
plt.scatter(coinc_eners.data()["energy"], smoothed_eners.data()["energy"])

ax.set_xlabel("Raw waveform energy")
ax.set_ylabel("Smoothed waveform energy")

plt.show()
    


print("Test completed!")
sys.exit()
