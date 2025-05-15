import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
sys.path.append("/Users/akannan/Downloads/Lab/Nab-Analysis")
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



run = Nab.DataRun(paths[2], 8118)
parameters = run.parameterFile()


noise = run.noiseWaves()
noise_headers = noise.headers()
#regcoinc = run.coincWaves()
#coinc = run.coincWaves().headers()

# Filter for populated pixel (1061) indices
singles_indices = singles_headers[singles_headers['pixel'] == 1061].index.tolist()

print(f'Number of waveforms: {len(noise.waves())}')

noise_list = []
for i in range(len(noise.waves())):
    noise_list.append(noise.wave(i).tolist())

#print(len(noise_list))
#print(len(noise_list[204]))

    
with open("NabData.JSON", 'r') as f:
    data_dict = json.load(f)

data_dict["Background"] = noise_list

with open("NabData.JSON", 'w') as f:
    json.dump(data_dict, f)

print("Test completed!")
sys.exit()




#------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------




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


print("TEST COMPLETED!")
