#%%
import sys
from nab_noise_reduction import NabFalseProtons
import time
import matplotlib.pyplot as plt
import random
import numpy as np
start = time.time()
#%%
analyzer = NabFalseProtons(user='arush', run_number=7616)
#%%
protons = analyzer.extract_protons()
params, params_covariances = protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100), function = analyzer.double_gaussian, parameterNames=analyzer.parameterNames)
plt.clf()


plt.figure(figsize=(10, 6))
plt.title("protons hits vs energy (DAQ)")
protons.hist('energy', bins = analyzer.Nab.np.arange(0,100))
protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100), function = analyzer.double_gaussian, parameterNames=analyzer.parameterNames)
#%%

print(params)

bound = analyzer.noise_gaussian_SD_bound(params, 3)
print(f'3sds from noise peak {bound}')

protons.defineCut("energy", '>=', bound)
plt.figure(figsize=(10, 6))
plt.title("protons hits vs energy (bounded filtering)")
protons.hist('energy', bins = analyzer.Nab.np.arange(0,100))
params2, cov2_par = protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100))
print(params2)

print('Completed bounded filter of protons. Moving on to eliminating viable electrons')
'''
lowerbound = analyzer.double_gaussian_turning_point(params)
print(f'noise and curve intersection {lowerbound}')

protons.defineCut("energy", '>=', lowerbound)
plt.figure(figsize=(10, 6))
plt.title("protons hits vs energy (bounded filtering)")
protons.hist('energy', bins = analyzer.Nab.np.arange(0,100))
protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100))
'''

#%%

electrons = analyzer.extract_electrons()

print("\n--- Filtering Electrons by Time ---")
stats = analyzer.filter_electrons_by_time_range(window=10000, lower_bound=0)
associations = analyzer.associate_protons_with_electrons(window=10000, lower_bound=2000)

print(f"Initial: {stats['initial']}, Filtered: {stats['filtered']}, Removed: {stats['removed']}")
#%%

print(associations)

#%%
electron_counts = [len(e_idxs) for e_idxs in associations.values()]
plt.figure(figsize=(10, 6))
counts, bin_edges, _ = plt.hist(electron_counts, bins=1000, color='skyblue', edgecolor='black')
plt.title('Histogram of Electron Counts per Proton')
plt.xlabel('Number of Electrons Associated with a Proton')
plt.ylabel('Number of Protons')
max_y = int(max(counts)) + 1
max_x = int(max(bin_edges)) + 1
plt.yticks(np.arange(0, max_y + 1, step=max(1, max_y // 10)))
plt.xticks(np.arange(0, max_x + 1, step=max(1, max_x // 20)))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#%%

proton_indices = list(analyzer.Ptimestamp.keys())
electron_indices = list(analyzer.Etimestamp.keys())

Ppixels = {}
Epixels = {}

analyzer.coinc.resetCuts()
analyzer.coinc.defineCut("custom", proton_indices)

protonP = analyzer.coinc.determineEnergyTiming(method='trap', params=analyzer.filter_settings)
indices = analyzer.coinc.headers().index.tolist()
for i in range(len(indices)):
    h = np.array(analyzer.coinc.head(i, pandas=True))
    Ppixels[indices[i]] = h[12]

print(Ppixels)
#%%
analyzer.coinc.resetCuts()
analyzer.coinc.defineCut("custom", electron_indices)

electronP = analyzer.coinc.determineEnergyTiming(method='trap', params=analyzer.filter_settings)
indices = analyzer.coinc.headers().index.tolist()
for i in range(len(indices)):
    h = np.array(analyzer.coinc.head(i, pandas=True))
    Epixels[indices[i]] = h[12]

print(Epixels)



#%%

LdetEs = [
    Eidx for Eidx, Epx in Epixels.items()
    if Epx >= 1 and Epx<129
]

LdetPs = [
    Pidx for Pidx, Ppx in Ppixels.items()
    if Ppx >= 1 and Ppx<129
]
UdetEs = [
    Eidx for Eidx, Epx in Epixels.items()
    if Epx >= 1001 and Epx<1129
]

UdetPs = [
    Pidx for Pidx, Ppx in Ppixels.items()
    if Ppx >= 1001 and Ppx<1129
]

#%%
# Loop through each proton and its associated electrons
for proton_idx, electron_idxs in associations.items():
    # Determine the proton's pixel position
    proton_pixel = Ppixels.get(proton_idx, None)
    
    if proton_pixel is None:
        continue  # Skip if no pixel data is available for this proton
    
    # Check if the proton is in one of the detector groups (LdetPs or UdetPs)
    if proton_pixel in Ppixels.values():
        if proton_pixel in LdetPs:  # Proton is in LdetPs
            # Remove electrons not in UdetEs
            new_electron_idxs = [
                e_idx for e_idx in electron_idxs if Epixels.get(e_idx, -1) in UdetEs
            ]
        elif proton_pixel in UdetPs:  # Proton is in UdetPs
            # Remove electrons not in LdetEs
            new_electron_idxs = [
                e_idx for e_idx in electron_idxs if Epixels.get(e_idx, -1) in LdetEs
            ]
        else:
            continue  # Skip if the proton is not in either detector
        
        # Update the association for this proton
        associations[proton_idx] = new_electron_idxs
    else:
        # Skip if the proton pixel data is not found
        continue

#%%
print("Hopefully these are the filtered count of electrons")
electron_counts = [len(e_idxs) for e_idxs in associations.values()]
plt.figure(figsize=(10, 6))
counts, bin_edges, _ = plt.hist(electron_counts, bins=1000, color='skyblue', edgecolor='black')
plt.title('Histogram of Electron Counts per Proton')
plt.xlabel('Number of Electrons Associated with a Proton')
plt.ylabel('Number of Protons')
max_y = int(max(counts)) + 1
max_x = int(max(bin_edges)) + 1
plt.yticks(np.arange(0, max_y + 1, step=max(1, max_y // 10)))
plt.xticks(np.arange(0, max_x + 1, step=max(1, max_x // 20)))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



#%%
#------------------------------------------------------------------------------

#%%

'''
from collections import defaultdict

# Step 1: Count how many protons each electron is associated with
electron_counts = defaultdict(int)
for electron_list in associations.values():
    for e_idx in electron_list:
        electron_counts[e_idx] += 1

# Step 2: Build the filtered dictionary with only unique (1-to-1) associations
uniquePEs = {}

for p_idx, e_list in associations.items():
    filtered_electrons = [e_idx for e_idx in e_list if electron_counts[e_idx] == 1]
    if filtered_electrons:
        uniquePEs[p_idx] = filtered_electrons
'''

