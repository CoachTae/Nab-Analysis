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


run_num = 8053


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
run = Nab.DataRun(paths[2], run_num)
print(f'File loaded: {time.time() - start_time} seconds')

# Load coincidence waveforms
noise = run.noiseWaves()
noise_headers = noise.headers().sort_values(by='pixel')

# i is our pixel number
with open("Pixel Noise Profiles.txt", 'w') as f:
    f.write(f"Run {run_num}\n")
    f.write("Pixel\t# Waveforms\tMean\tSD\n")
    for i in range(128):
        noise_vals = None
        # Find which waveform ids are associated with the current pixel
        WFids = noise_headers[noise_headers['pixel'] == i].index.tolist()

        # Combine all noise waveforms into one
        for WFid in WFids:
            try:
                noise_vals += noise.wave(WFid)
            except:
                noise_vals = noise.wave(WFid)

        try:
            f.write(f"{i+1}\t{len(WFids)}\t{round(np.mean(noise_vals),2)}\t{round(np.std(noise_vals),2)}\n")
        except:
            f.write(f"{i+1}\t{len(WFids)}\tNone\tNone\n")



    


print("Test completed!")
sys.exit()
