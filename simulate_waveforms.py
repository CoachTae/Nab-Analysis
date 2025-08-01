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


sample = noise.wave(0)
t1 = np.arange(len(sample))
plt.figure(figsize=(12, 6))
plt.plot(t1, sample, label="Raw Waveform")
plt.title(f"Noise {1}")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


#%%
def sim_wf(t0, amplitude, rise_time, decay_tau, dt=0.01, duration=None):
    """
    Simulate a waveform with linear rise and exponential decay.

    Parameters:
        t0 (float): Start time of the waveform
        amplitude (float): Peak amplitude of the waveform
        rise_time (float): Time it takes to rise to the peak
        decay_tau (float): Exponential decay time constant
        dt (float): Time step (sampling interval)
        duration (float): Total simulation duration; if None, auto-estimated

    Returns:
        t (np.ndarray): Time array
        wf (np.ndarray): Waveform values
        energy (float): Integral (area under the curve)
    """
    if duration is None:
        duration = rise_time + 10 * decay_tau  # auto-extend if not set

    t = np.arange(t0, t0 + duration, dt)
    wf = np.zeros_like(t)

    # Linear rise
    rise_mask = (t >= t0) & (t < t0 + rise_time)
    wf[rise_mask] = amplitude * (t[rise_mask] - t0) / rise_time

    # Exponential decay
    decay_mask = t >= (t0 + rise_time)
    wf[decay_mask] = amplitude * np.exp(-(t[decay_mask] - (t0 + rise_time)) / decay_tau)

    # Energy = area under the waveform
    energy = np.trapz(wf, t)

    return t, wf, energy
#%%

t0=6000
amp = 110
risetime = 10
exp_decay_param = 350 

t, wf, energy = sim_wf(t0=t0, amplitude=amp, rise_time=risetime, decay_tau=exp_decay_param)
print(f"Energy: {energy:.4f}")
sample_resampled = np.interp(t, np.linspace(t1[0], t1[-1], len(t1)), sample)

wf = wf + sample_resampled

t_before = np.linspace(t1[0], t0, len(t1[:np.argmax(t1 >= t0)])) 
sample_before = sample[:np.argmax(t1 >= t0)]

t_overlap = t

sample_overlap = sample_resampled

t_after = np.linspace(t[-1], t1[-1], len(t1[t1 >= t[-1]]))
sample_after = sample[t1 >= t[-1]]

time_combined = np.concatenate([t_before, t_overlap, t_after])
waveform_combined = np.concatenate([sample_before, wf, sample_after])

plt.figure(figsize=(12, 6))

plt.plot(time_combined, waveform_combined)

plt.xlim(t1[0], t1[-1])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Waveform and Sample")
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()
    
    