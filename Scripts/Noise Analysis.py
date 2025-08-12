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



run = Nab.DataRun(paths[2], 7900)
parameters = run.parameterFile()

coinc = run.coincWaves()
coinc_headers = run.coincWaves().headers()
noise = run.noiseWaves()
noise_headers = run.noiseWaves().headers()

# Filter for electrons
coinc_indices = coinc_headers[coinc_headers['hit type'] == 2].index.tolist()

# Find pixel with highest electron count
#pixel_counts = coinc_headers['pixel'][coinc_indices].value_counts()


# Filter for electrons of the most populated pixel
coinc.defineCut("pixel", "=", 1097)
coinc.defineCut("hit type", "=", 2)

waves = []
for wave in coinc.waves():
    waves.append(wave)

print(len(waves))
