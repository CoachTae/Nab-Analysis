#%%
from nab_noise_reduction import NabFalseProtons
import time
import matplotlib.pyplot as plt
import random

start = time.time()
#%%
analyzer = NabFalseProtons(user='arush', run_number=7616)
#%%
print("\n--- Extracting Protons ---")
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
protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100))

params2, cov2_par = protons.fitHist('energy', bins = analyzer.Nab.np.arange(0,100))
print(params2)
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
#now comparing to timestamp:

electrons = analyzer.extract_electrons()
protons = analyzer.extract_protons()
analyzer.filter_protons_by_time_range()
false_ps, real_ps = analyzer.real_and_false_protons()


plt.figure(figsize=(10, 6))
plt.title("protons hits vs energy (timestamp filtering)")
real_ps.hist('energy', bins = analyzer.Nab.np.arange(0,100))
real_ps.fitHist('energy', bins = analyzer.Nab.np.arange(0,100), function = analyzer.double_gaussian, parameterNames=analyzer.parameterNames)


