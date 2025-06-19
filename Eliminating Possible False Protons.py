#%%
#1
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
import bisect
import filterpy
#%%
#2
'''Importing data'''

run = Nab.DataRun(paths[2], 7616)
parameters = run.parameterFile()

coinc = run.coincWaves() #Pulling coincidence waveforms
# Defining filter settings to be used throughout:
filter_settings = [1250, 50, 1250] 
#%%
#3
'''Identifying the timestamps of the electron hits'''
coinc.resetCuts()
# Pulling waveforms that the DAQ consideres electron coincidence waveforms:
coinc.defineCut("hit type", "=", 2) 

# Identifying data for these electrons:
elecE = coinc.determineEnergyTiming(method='trap', params=filter_settings)
print(f' Electrons \n {elecE.data()}') 
# Pulling a list of the indices of these electrons:
eheaders = coinc.headers()
indices = eheaders.index.tolist()
#Putting the timestamps of the electrons in a dictionary, with the keys being 
#the indices of the electrons
Etimestamp = {}
for i in range(len(indices)): 
    headers = np.array(coinc.head(i, pandas=True))
    Etimestamp[indices[i]] = headers[2]

presizeE = len(Etimestamp)
#%%
#4
'''Identifying the timestamps of the POTENTIAL proton hits'''
coinc.resetCuts()
# Pulling waveforms that the DAQ consideres proton coincidence waveforms:
coinc.defineCut("hit type", "=", 0)
# Identifying data for these protons:
protonE = coinc.determineEnergyTiming(method='trap', params=filter_settings)

totprotons = len(coinc.headers().index.tolist())
#%%
#5
# Ignoring the protons whose energies cannot be determined, or are non-physical
protonE.defineCut("energy", '>' , 0) # change to <= to see nonphysical wfs. 
cut = protonE.returnCut()
coinc.defineCut("custom", cut)

protonE = coinc.determineEnergyTiming(method='trap', params=filter_settings)
print(f'\n Protons \n {protonE.data()}') 
# Pulling a list of the indices of these protons:
indices = coinc.headers().index.tolist()
presizeP = len(indices)
#Putting the timestamps of the protons in a dictionary, with the keys being 
#the indices of the protons
Ptimestamp = {}
for i in range(len(indices)): 
    headers = np.array(coinc.head(i, pandas=True))
    Ptimestamp[indices[i]] = headers[2]



print(f'\n There are {presizeE} electrons identified by the DAQ')
print(f'\n There are {totprotons} protons identified by the DAQ')
print(f'\n There are {presizeP} protons with physical energies \n')
#%%
#6
'''Since the timestamps are in increments of 4 ns, and assuming the coincidence
range of 8000 - 40,000 ns, protons must be at most 10,000 timestamps from an 
electron. So we will check which possible protons fall outside of that range'''

# Sort the Etimestamp values
sorted_Etimestamps = sorted(Etimestamp.items(), key=lambda x: x[1])  # Sorting by the timestamp value
sorted_Ets = [ts for idx, ts in sorted_Etimestamps]  # Extract sorted electron timestamps
sorted_Eidx = [idx for idx, ts in sorted_Etimestamps]  # Extract corresponding indices

not_in_range = {}

# Iterate over a copy of the Ptimestamp items
for Pidx, Pts in list(Ptimestamp.items()):  # Create a list of items for safe iteration
    # Use binary search to find the nearest position in sorted electron timestamps
    pos = bisect.bisect_left(sorted_Ets, Pts - 10000)  # Find the first electron timestamp >= Pts - 10000

    # Check if there's an electron timestamp within the range [Pts - 10000, Pts + 10000]
    in_range = False
    while pos < len(sorted_Ets) and sorted_Ets[pos] <= Pts + 10000:
        if abs(sorted_Ets[pos] - Pts) <= 10000:
            in_range = True
            break
        pos += 1

    # If no match found, add to the not_in_range dictionary and remove from Ptimestamp
    if not in_range:
        not_in_range[Pidx] = Pts
        del Ptimestamp[Pidx] 

# This has no impact on the code, is simply present for verification purposes
postsizeP = len(Ptimestamp)
removedsizeP = len(not_in_range)
print(f"Inital proton counts: {presizeP}")
print(f"Updated proton counts: {postsizeP}")
print(f"Protons outside of 4 us of electron times: {removedsizeP}")
#%%
#7
'''Now finding the proton waveforms outside of the desired time range'''

falsePs = list(not_in_range.keys())
coinc.resetCuts()
coinc.defineCut("custom", falsePs)
poss_noise = coinc.determineEnergyTiming(method='trap', params=filter_settings)
print(f'\n Potential False Protons \n {poss_noise.data()}')
#%%
#8
'''Manually checking some protons'''
coinc.resetCuts()
i = random.choice(falsePs)

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

sys.exit()






























