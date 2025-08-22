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



run = Nab.DataRun(paths[2], 7900)
#parameters = run.parameterFile()

coinc = run.coincWaves()
#coinc_headers = run.coincWaves().headers()
noise = run.noiseWaves()
#noise_headers = run.noiseWaves().headers()

#coinc.defineCut('hit type', '=', 2)
#coinc.defineCut('pixel', '=', 1097)

# Count how many waveforms we can make
num_wfs = len(noise.waves())

# Make that many waveforms
wfs = SimWF.batch_wfs(3500, "random", "random", 1250,
                      ampmin=50, ampmax=750,
                      rise_timemin=8, rise_timemax=20,
                      N=num_wfs)

# Calculate "perfect" energies
real_energies, _ = bf.applyDoubleTrapFilter(wfs, 1250, 100, 1250)

# Add experimental noise
for i, wf in enumerate(wfs):
    if len(noise.wave(i)) > 7000:
        noise_wf = noise.wave(i)[:7000]
        wf += noise_wf
    else:
        wf += noise.wave(i)

# Find energies with noise
noise_energies, _ = bf.applyDoubleTrapFilter(wfs, 1250, 100, 1250)

# Filter out the noise
from Classes import KalmanFilterClass as KFClass

KF = KFClass.KF()

for i, wf in enumerate(wfs):
    wfs[i] = KF.smooth(wf)

filtered_energies, _ = bf.applyDoubleTrapFilter(wfs, 1250, 100, 1250)

with open("Noise Filter Effects on Energy.txt", 'w') as f:
    f.write("Real Energy\tNoised Energy\tFiltered Energy\tNoised Diff\tFiltered Diff\n")
    for i in range(len(wfs)):
        noised_diff = (real_energies[i] - noise_energies[i])*100/real_energies[i]
        filtered_diff = (real_energies[i] - filtered_energies[i])*100/real_energies[i]
        f.write(f'{round(real_energies[i],3)}\t{round(noise_energies[i],3)}\t{round(filtered_energies[i],3)}\t{round(noised_diff,3)}%\t{round(filtered_diff,3)}%\n')
