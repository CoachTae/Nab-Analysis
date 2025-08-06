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


dataLoc = '/Users/akannan/Downloads/'
num = 7616
dataFile = Nab.File(f'{dataLoc}Run{num}_1.h5')
run = Nab.DataRun(paths[2], 7616)
parameters = run.parameterFile()


coinc = run.coincWaves()
noise = run.noiseWaves()



kwargs = {'alpha': 0.75,'cmap': 'plasma','logNorm': True}
kwargsPlot = {'labels':np.asarray(np.arange(1,128),dtype=str), 'labelValues': True}
fig = dataFile.plotHitLocations(plot = True, sourceFile='noise', kwargsFig = kwargs, kwargsPlot=kwargsPlot)

fig



#%%
headers = noise.headers()
indices = headers.index.tolist()

noise.resetCuts()
i = random.choice(indices)
s1 =  noise.wave(i)
i = random.choice(indices)
s2 = noise.wave(i)
i = random.choice(indices)
s3 = noise.wave(i)

st = np.append(s1, s2)
stt = np.append(st, s3)


avg = np.mean(stt)
std = np.std(stt)

print(std)


#%%
i=random.randint(0, 200)
w1=  coinc.wave(i)
t1 = np.arange(len(w1))
plt.figure(figsize=(12, 6))
plt.plot(t1, w1, label="Raw Waveform")
plt.title(f"Coinc {i}")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


#%%
def sim_wf(t0, amplitude, rise_time, decay):
    """
    Simulate a waveform with fixed time base from 0 to 14000.
    The waveform rises linearly to a set energy then decays exponentially.

    Parameters:
        t0 (float): Time of start of waveform (in samples)
        amplitude (float): Peak height
        rise_time (float): Time to reach the peak (in samples)
        decay (float): Decay time constant (in samples)

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
        wf[i] = amplitude * np.exp(-(i - rise_end) / decay)

    # Energy = sum since dt = 1
    energy = np.sum(wf)

    return t, wf, energy

def gaussian_noise(mean=0.0, std=1.0):
    """
    Generate Gaussian (normal) noise.

    Parameters:
        length (int): Number of samples (typically 14001 for your waveform)
        mean (float): Mean of the noise
        std (float): Standard deviation of the noise

    Returns:
        noise (np.ndarray): Noise array of length `length`
    """
    return np.random.normal(loc=mean, scale=std, size=14000)
#%%

noise = gaussian_noise(mean=avg, std=std)

t0=6000
amp = 40
risetime = 10
exp_decay_param = 2000

t, wf, energy = sim_wf(t0=t0, amplitude=amp, rise_time=risetime, decay=exp_decay_param)

waveform = wf + noise



plt.figure(figsize=(12, 6))
plt.plot(t, waveform)
plt.xlim(t[0], t[-1])
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Simulated Waveform")
plt.grid(True)
plt.legend()
plt.tight_layout()








   