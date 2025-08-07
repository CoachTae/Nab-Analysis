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
from scipy.optimize import curve_fit


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




def safe_exp(x, max_exp=500):
    # Avoid overflow in exp by capping it
    x = np.clip(x, -max_exp, max_exp)
    return np.exp(x)


def curve_func(x, b, m, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12):
    z1 = ((x - P2)/P3)
    z2 = ((x - P8)/P9)

    F1 = P1*safe_exp(-(1/2)*(z1**2))
    F2 = P6/(1 + safe_exp(z1))**2
    F3 = P4*(safe_exp(P5*z1))/(1 + safe_exp(z1))**4
    
    F4 = P7*safe_exp(-(1/2)*(z2**2))
    F5 = P12/(1 + safe_exp(z2))**2
    F6 = P10*(safe_exp(P11*z2))/(1 + safe_exp(z2))**4

    result = F1 + F2 + F3 + F4 + F5 + F6 + m*x + b


    # Check for non-finite output
    if not np.all(np.isfinite(result)):
        print("Non-finite result detected!")
        print(f'z1: {z1}')
        print(f'z2: {z2}')
        print(f"Params: {[b, m, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12]}")
        print(f"Result: {result}")
        sys.exit()
    
    return result







run = Nab.DataRun(paths[2], 8118)
parameters = run.parameterFile()


#noise = run.noiseWaves()
#regcoinc = run.coincWaves()
#coinc = run.coincWaves().headers()
singles = run.singleWaves()
singles_headers = run.singleWaves().headers()

# Filter for populated pixel (1061) indices
singles_indices = singles_headers[singles_headers['pixel'] == 1061].index.tolist()

# Take every 100th waveform (since applying Kalman filter is time intensive)
#singles_indices = [idx for i, idx in enumerate(singles_indices) if i % 100 == 0]

# Cut for only waveforms from most populated pixel
singles.defineCut('custom', singles_indices)

'''# Initialize filter
KFilter = KF()

# Apply filter to selected waveforms
numwaves = len(singles_indices)
counter = 0
for index in singles_indices:
    singles._waveformFile__waves[index] = KFilter.smooth(singles._waveformFile__waves[index])
    counter += 1
    if counter % 100 == 0:
        print(f'Time elapsed: {time.time() - start_time})')
        print(f'{counter*100/numwaves}% ({counter} out of {numwaves}\n')'''

# Apply trap filter and determine energies
filter_settings = [1250, 50, 1250]
singles_energies = singles.determineEnergyTiming('trap', params=filter_settings)

# Get headers object w/ energy column
energies_headers = singles_energies.data()

# Pull just the energies
energies = energies_headers['energy']*0.3



# Histogram
bin_size = 0.5
min_energy = energies.min()
max_energy = energies.max()
bins = np.arange(min_energy, max_energy + bin_size, bin_size)


counts, bin_edges = np.histogram(energies, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


p0 = [
    800,  # P1: amplitude of first Gaussian
    50,   # P2: center of first Gaussian
    5,    # P3: width of first Gaussian
    400,  # P4: amplitude for F3
    0.2,  # P5: shape param
    200,  # P6: amplitude for F2
    0,    # m: slope (try 0 to start)
    200,  # P7: amplitude of second Gaussian
    100,  # P8: center of second Gaussian
    5,    # P9: width of second Gaussian
    100,  # P10: amplitude for F6
    0.2,  # P11: shape param for F6
    80,   # P12: amplitude for F5
    0    # b: background
]


lower_bounds = [0, 0, 0.1, 0, -10, 0, 0, 0, 0, 0.1, 0, -10, 0, 0]
upper_bounds = [1e4, 200, 50, 1e4, 10, 1e4, 1, 1e4, 200, 50, 1e4, 10, 1e4, 1000]

popt, pcov = curve_fit(curve_func, bin_centers, counts, p0=p0, bounds=(lower_bounds, upper_bounds))

plt.hist(energies, bins=bins, alpha=0.5, label="Data")
plt.plot(bin_centers, curve_func(bin_centers, *popt), 'r-', label='Fit')
plt.legend()
plt.xlabel('Energy')
plt.ylabel('Count')
plt.show()


print("TEST COMPLETED!")
