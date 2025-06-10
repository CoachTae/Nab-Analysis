import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
sys.path.append('/Users/akannan/Downloads/Lab/Nab-Analysis')
import Paths

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
import filterpy

run = Nab.DataRun(paths[2], 5730)
parameters = run.parameterFile()

coinc = run.coincWaves()

'''Identifying the timestamps of the electron hits'''

coinc.defineCut("hit type", "=", 0)

filter_settings = [1250, 50, 1250]

elecE = coinc.determineEnergyTiming(method='trap', params=filter_settings)

#print(elecE.data())

Etimestamp = {}
N = len(coinc.waves())
for i in range(N): 
    headers = np.array(coinc.head(i, pandas=True))
    Etimestamp[i] = headers[2]

'''Identifying the timestamps of the POTENTIAL proton hits'''

coinc.resetCuts()
coinc.defineCut("hit type", "=", 2)

protonE = coinc.determineEnergyTiming(method='trap', params=filter_settings)

#print(protonE.data())

Ptimestamp = {}
N = len(coinc.waves())
for i in range(N): 
    headers = np.array(coinc.head(i, pandas=True))
    Ptimestamp[i] = headers[2]
# This has no impact on the code, is simply present for verification purposes
presizeP = len(Ptimestamp)

'''Since the timestamps are in increments of 4 ns, and assuming the coincidence
range of 8000 - 40,000 ns, protons must be at most 10,000 timestamps from an 
electron. So we will check which possible protons fall outside of that range'''

not_in_range = {}

for key, item in list(Ptimestamp.items()):
    # Check if the item is within 10,000 timestamps of any electron timestamp
    in_range = False
    for electron_key, dict_value in Etimestamp.items():
        if abs(item - dict_value) <= 15000:
            in_range = True
            break  # Exit the loop as soon as a match is found
    
    if not in_range:
        not_in_range[key] = item  # Add the item to not_in_range dictionary
        del Ptimestamp[key]  # Remove the item from Ptimestamp dictionary

# This has no impact on the code, is simply present for verification purposes
postsizeP = len(Ptimestamp)
removedsizeP = len(not_in_range)
print(f"Inital proton counts: {presizeP}")
print(f"Updated proton counts: {postsizeP}")
print(f"Values outside of 4 us of electron times: {removedsizeP}")

'''Now finding the proton waveforms outside of the desired time range'''

nir_idx = list(not_in_range.keys())
coinc.resetCuts()
coinc.defineCut("custom", nir_idx)
poss_noise = coinc.determineEnergyTiming(method='trap', params=filter_settings)
#print(poss_noise.data())

'''Attempting to plot these possible protons'''

j = len(coinc.waves())

coinc.resetCuts()
coinc.defineCut("hit type", "=", 2)
loopcount=0
for k in range(len(nir_idx)):

    i = nir_idx[k]
    sample = coinc.wave(i)
    t = np.arange(len(sample))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(t, sample, label='Raw Waveform')
    ax.legend()
    ax.set_title(f'Proton {i}')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    loopcount += 1
    
print(loopcount)

sys.exit()






pix_n = 4
coinc.defineCut('pixel', '=', pix_n)
coinc.defineCut("hit type", "=", 0)

#print(f'Number of waveforms: {len(noise.waves())}')
#print(f'Initial # of particle counts: {len(coinc.waves())}')

N = len(coinc.waves())
i = random.randint(0,N-1)
sample = coinc.wave(i)

t = np.arange(len(sample))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,6))
axs[0].plot(t, sample, label='Raw Waveform')
axs[0].legend()
#axs[1].plot(t, filtered, label='Smoothed Waveform', color='orange')
#axs[1].legend()

sys.exit()

#noise = run.noiseWaves()
regcoinc = run.coincWaves()
coinc = run.coincWaves().headers()
print(regcoinc.headerType)
print(f'Finished pulling waveforms: {time.time() - start_time} seconds')
print(len(regcoinc.waves()))

filter_settings = [1250, 50, 1250]
regcoinc.defineCut("hit type", "=", 2)
print(len(regcoinc.waves()))
coinc_energies = regcoinc.determineEnergyTiming(method='trap', params=filter_settings, batchsize=10)
print(coinc_energies.data())
coinc_energies.resetCuts()
coinc_energies.defineCut("energy", "between", 100, 3000)

temp = coinc_energies.data()
electron_E = temp["energy"]

print(electron_E)

print("Test completed!")


#------------------------------------------------------------------------------

''' Visualizing noise on nth pixel. Pixel chosen from hit map '''

pix_n = 32
 
run = Nab.DataRun(paths[2], 5730)
parameters = run.parameterFile()
noise = run.noiseWaves()


noise.resetCuts()
noise.defineCut('pixel', '=', pix_n)

spectra = noise.generatePowerSpectra()

Nab.plt.plot(spectra[0][1:], spectra[1][1:])
Nab.plt.yscale('log')
Nab.plt.xscale('log')
Nab.plt.xlabel('Frequency (Hz)')
Nab.plt.ylabel('ADC^2/Hz')
Nab.plt.title('Power Spectra for Pixel 35')
Nab.plt.show()

print("Test completed!")

#------------------------------------------------------------------------------
# Load in data
data = Nab.File(paths[2]+"Run7597_0.h5")



# Extract coincidence waveforms
coinc = data.coincWaves()
pulsr = data.pulsrWaves()
noise = data.noiseWaves()
single = data.singleWaves()

'''params: list with varying elements depending on the method passed
        (optional parameters shown in parenthesis)
        'trap': [risetime, flat top length, decay rate, (threshold percent, mean, shift)]
        'cusp': [risetime, flat top length, decay rate, (threshold percement, mean, shift)]
        'doubletrap': [risetime, flat top length, decay rate, (threshold percent, mean, shift)]'''




filter_settings = [1250, 50, 1250]

# Coincidence waves energy timings
Ctimings = coinc.determineEnergyTiming(method='trap', params=filter_settings)
print(Ctimings)

# Plotting energies of coincidence waves
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('xkcd:white')
fig.text(0.5, -0.05, "Generated with: coinc.determineEnergyTiming(method='trap', params=filter_settings)", ha='center')
ax.set_xlabel('ADC Channel')
ax.set_ylabel('Counts')
ax.grid(True)
ax.set_title('Energy Histogram \n Singles Data')
Ctimings.hist('energy', bins = Nab.np.arange(0,6000))
plt.xlim(0,6000)
plt.ylim(0,20)
Ctimings.data().columns
plt.show()





#------------------------------------------------------------------------------
# Print something to show that the code has resolved completely.
print("Test successful!")