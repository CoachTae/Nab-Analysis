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


run_num = 7616
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



def plot_comparison(sample, smoothed):
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




# Load Data
run = Nab.DataRun(paths[2], run_num)
print(f'File loaded: {time.time() - start_time} seconds')


# Load waveforms
coinc = run.coincWaves()
coinc_header = coinc.headers()
noise = run.noiseWaves()
noise_header = noise.headers()
print(f'Waveforms extracted.')


# Map waveform indices to pixel number
wf_to_pixel = {} # Dictionary. Keys = waveform id/#, Values = pixel number
pixels = []
for i, pixel in enumerate(coinc_header['pixel']):
    if coinc_header.iloc[i]['hit type'] == particle_type:
        wf_to_pixel[i] = pixel
        if pixel not in pixels:
            pixels.append(pixel)

pixels.sort()

means = []
sds = []
# Get noise information for each pixel
for i in pixels:
    noise_vals = None
    
    WFids = noise_header[noise_header['pixel'] == i].index.tolist()

    for WFid in WFids:
        try:
            noise_vals += noise.wave(WFid)
        except:
            noise_vals = noise.wave(WFid)


    try:
        means.append(np.mean(noise_vals))
        sds.append(np.std(noise_vals))
    except:
        means.append(None)
        sds.append(None)


avg_mean = np.mean(means)
avg_sd = np.mean(sds)
for i in range(len(means)):
    if means[i] is None:
        means[i] = avg_mean
    else:
        pass

    if sds[i] is None:
        sds[i] = avg_sd
    else:
        pass
    
pixel_to_noise = {pixel: (mean, sd) for pixel, mean, sd in zip(pixels, means, sds)}

# Select 10 random waveforms
random_WFs = []
for i in range(10):
    choice = random.choice(list(wf_to_pixel.keys()))
    while True:
        if choice in random_WFs:
            choice = random.choice(list(wf_to_pixel.keys()))
        else:
            break
    random_WFs.append(choice)


# Apply cuts so we don't calculate every energy
coinc.defineCut('custom', random_WFs)

# Determine raw energy
raw_energy_timing = coinc.determineEnergyTiming(method='trap', params=[1250, 50, 1250], useGPU=True, batchsize=100)
raw_energies = raw_energy_timing.data()['energy']
print(raw_energies)

# Create filter
KFilter = KF()


# Open up waveforms for modification
for index in random_WFs:
    WF = coinc._waveformFile__waves[index]
    pixel = wf_to_pixel[index]
    mean, sd = pixel_to_noise[pixel]

    WF -= mean
    KFilter.set_observation_covariance(sd)

    smoothed_WF = KFilter.smooth(WF)

    coinc._waveformFile__waves[index] = smoothed_WF

    # Uncomment this line if you want to plot each before-and-after waveform
    #plot_comparison(WF, smoothed_WF)

filtered_energy_timing = coinc.determineEnergyTiming(method='trap', params=[1250, 50, 1250], useGPU=True, batchsize=100)
filtered_energies = filtered_energy_timing.data()['energy']


print(filtered_energies)

    


print("Test completed!")
sys.exit()
