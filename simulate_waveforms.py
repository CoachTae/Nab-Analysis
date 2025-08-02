import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
sys.path.append('/Users/akannan/Downloads/Lab/Nab-Analysis')
import Paths
from scipy.interpolate import interp1d

start_time = time.time()
User = 'arush'

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

#%%


run = Nab.DataRun(paths[2], 7616)
parameters = run.parameterFile()


coinc = run.coincWaves()
noise = run.noiseWaves()

noise.resetCuts()
noise.defineCut("pixel", "=",  12)


w1=  noise.wave(0)
t1 = np.arange(len(w1))
plt.figure(figsize=(12, 6))
plt.plot(t1, w1, label="Raw Waveform")
plt.title(f"Noise {1}")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


#%%
def sim_wf(t0, amplitude, rise_time, decay_tau):
    """
    Simulate a waveform with fixed time base from 0 to 14000 with dt=1.
    The waveform rises linearly to `amplitude` then decays exponentially.

    Parameters:
        t0 (float): Time of start of waveform (in samples)
        amplitude (float): Peak height
        rise_time (float): Time to reach the peak (in samples)
        decay_tau (float): Decay time constant (in samples)

    Returns:
        t (np.ndarray): Time array from 0 to 14000
        wf (np.ndarray): Waveform values, length 14001
        energy (float): Integral (sum) of waveform
    """
    t = np.arange(0, 14000)  # fixed time base
    wf = np.zeros_like(t, dtype=np.float64)

    t0 = int(t0)
    rise_end = t0 + int(rise_time)

    # Linear rise
    for i in range(t0, min(rise_end, len(wf))):
        wf[i] = amplitude * (i - t0) / rise_time

    # Exponential decay
    for i in range(rise_end, len(wf)):
        wf[i] = amplitude * np.exp(-(i - rise_end) / decay_tau)

    # Energy = sum since dt = 1
    energy = np.sum(wf)

    return t, wf, energy

def generate_gaussian_noise(mean=0.0, std=1.0, seed=None):
    """
    Generate Gaussian (normal) noise.

    Parameters:
        length (int): Number of samples (typically 14001 for your waveform)
        mean (float): Mean of the noise
        std (float): Standard deviation of the noise
        seed (int or None): Optional random seed for reproducibility

    Returns:
        noise (np.ndarray): Noise array of length `length`
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=14000)
#%%

noise = generate_gaussian_noise(mean=8.759, std=3.1439)

t0=6000
amp = 110
risetime = 10
exp_decay_param = 350 

t, wf, energy = sim_wf(t0=t0, amplitude=amp, rise_time=risetime, decay_tau=exp_decay_param)

waveform = wf + noise



plt.figure(figsize=(12, 6))
plt.plot(t, waveform)
plt.xlim(t[0], t[-1])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform and Sample")
plt.grid(True)
plt.legend()
plt.tight_layout()
   