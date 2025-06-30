#%%
from nab_noise_reduction import NabFalseProtons
import time
import random

start = time.time()

analyzer = NabFalseProtons(user='arush', run_number=7616)
#%%
print("\n--- Extracting Electrons ---")
electrons = analyzer.extract_electron_timestamps()
print(electrons.data())
#%%
print("\n--- Extracting Protons ---")
protons = analyzer.extract_proton_timestamps()
print(protons.data())
#%%
print("\n--- Filtering Protons by Time ---")
stats = analyzer.filter_protons_by_time_range()
print(f"Initial: {stats['initial']}, Filtered: {stats['filtered']}, Removed: {stats['removed']}")
#%%
print("\n--- Separating Real and False Protons ---")
false_ps, real_ps = analyzer.determine_real_and_false_protons()
print("\nFalse Protons:\n", false_ps.data())
print("\nReal Protons:\n", real_ps.data())
#%%
print("\n--- Plotting a Random False Proton ---")

if analyzer.not_in_range:
    random_idx = random.choice(list(analyzer.not_in_range.keys()))
    analyzer.plot_waveform(random_idx)

print(f"\n Done in {time.time() - start:.2f} seconds")