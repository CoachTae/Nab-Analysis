import os
import sys
import time
import random
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


'''
This script does the following:
    Loads a file specified on line 47
    Picks a random proton/electron (specified on line 58 by 0 or a 2)
    Plots 2 plots on top of each other
        Top plot is the raw waveform
        Bottom plot is the smoothed waveform
'''


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



# This "try" statement is for users using WSL which has no graphical display
# On normal systems or with systems without tkinter, it will just pass instead.
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass # Fallback to system defaults

run = Nab.DataRun(paths[2], 7616)
parameters = run.parameterFile()


#noise = run.noiseWaves()
#noise_headers = noise.headers()
coinc = run.coincWaves()
#coinc_headers = coinc.headers()

coinc.defineCut("hit type", "=", particle_type)

#print(f'Number of waveforms: {len(noise.waves())}')
#print(f'Initial # of particle counts: {len(coinc.waves())}')

N = len(coinc.waves())

KFilter = KF()
KFilter.set_transition_covariance(0.001)
KFilter.set_observation_covariance(5)

while True:

    i = random.randint(0, N - 1)



    sample = coinc.wave(i)
    smoothed = KFilter.smooth(sample)


    #print(f'Sample shape: {sample.shape}')
    #print(f'Smoothed shape: {smoothed.shape}')


    t = np.arange(len(sample))

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,6))

    axs[0].plot(t, sample, label='Raw Waveform')
    axs[0].legend()

    axs[1].plot(t, smoothed, label='Smoothed Waveform', color='orange')
    axs[1].legend()

    ymin, ymax = axs[0].get_ylim()

    axs[0].set_ylim(ymin, ymax)
    axs[1].set_ylim(ymin, ymax)

    plt.xlabel('Time (us)')
    plt.tight_layout()
    plt.show()
    


print("Test completed!")
sys.exit()
